"""Model-side explanations: which neighbours and edges the served R-GCN relied on for a drug pair.

This is the GNNExplainer objective (Ying et al., 2019): learn a soft mask in [0, 1] over edges so that
the masked graph keeps the model's prediction, with penalties on mask size and mask entropy, so only
the edges the score actually depends on stay switched on.

Scope of the mask. The pair score is a function of h_a and h_b, the two drugs' final embeddings. We mask
every edge that feeds those two embeddings directly, in both R-GCN layers (the same mask in each layer,
as in PyG's GNNExplainer): knowledge-graph edges into A or B (enzymes, targets, side effects, classes,
indications ...) and the DDI-channel edges from their known interaction partners. The rest of the graph
(how a neighbour's own embedding was formed) is held fixed at its unmasked value. That keeps one
explanation under a second on CPU, small enough to run per API request.

Why not torch_geometric.explain directly: PyG's Explainer injects the mask inside MessagePassing, but
RGCNConv propagates one relation at a time (mask size mismatch) and the DDI channel is a dense matmul
outside message passing. So the masked forward below re-implements the two layers for the target nodes
and is checked to reproduce the model's own score exactly when every mask is 1.
"""
import numpy as np
import torch
import torch.nn.functional as F

REL_LABEL = {"enzyme": "enzyme", "target": "target", "transporter": "transporter", "carrier": "carrier",
             "side_effect": "side effect", "indication": "indication", "contraindication": "contraindication",
             "off_label": "off-label use", "in_class": "drug class", "ppi": "protein link", "ddi": "known interaction"}


@torch.no_grad()
def layer_states(model, drug_x, edge_index, edge_type, A_norm):
    """[x0, x1, x2]: node states before the first layer, after it, and after the second (eval mode)."""
    model.eval()
    x = torch.cat([model.drug_in(drug_x), model.node_emb.weight], 0)
    D = model.n_drugs
    states = [x]
    for conv, lin, norm in zip(model.convs, model.ddi_lin, model.norms):
        h = conv(x, edge_index, edge_type)
        h = torch.cat([h[:D] + lin(A_norm @ x[:D]), h[D:]], 0)
        x = x + F.relu(norm(h))
        states.append(x)
    return states


class ModelExplainer:
    def __init__(self, model, states, edge_index, edge_type, rel_names, A, n_drugs):
        """A: binary DDI adjacency (dense, n_drugs x n_drugs) used for the served embeddings."""
        assert model.conv_kind == "rgcn", "masked forward is implemented for R-GCN only"
        self.model, self.states, self.D = model, states, n_drugs
        self.rel_names = [r[4:] if r.startswith("rev_") else r for r in rel_names]
        self.n_rel = len(rel_names)
        self.A = A
        self.deg = A.sum(1).clamp(min=1.0)
        dst = edge_index[1].numpy()
        order = np.argsort(dst, kind="stable")
        self.src_sorted, self.type_sorted = edge_index[0].numpy()[order], edge_type.numpy()[order]
        self.ptr = np.searchsorted(dst[order], np.arange(len(states[0]) + 1))

    # ------------------------------------------------------------ candidate edges
    def _incoming(self, t):
        lo, hi = self.ptr[t], self.ptr[t + 1]
        src, typ = torch.from_numpy(self.src_sorted[lo:hi]).long(), torch.from_numpy(self.type_sorted[lo:hi]).long()
        partners = torch.nonzero(self.A[t] > 0).flatten()
        return src, typ, partners

    @torch.no_grad()
    def _messages(self, t, src, typ, partners):
        """Per-edge messages into target t for each layer. Neighbours' states are fixed, so these are
        constants; only the mask (and t's own layer-1 state) changes during the optimisation."""
        out = []
        cnt = torch.bincount(typ, minlength=self.n_rel).clamp(min=1).float()
        for k, x in enumerate(self.states[:2]):
            conv, lin = self.model.convs[k], self.model.ddi_lin[k]
            W = (conv.comp @ conv.weight.view(conv.num_bases, -1)).view(self.n_rel, conv.in_channels_l, -1)
            kg = torch.empty(len(src), W.shape[-1])
            for r in typ.unique().tolist():
                sel = typ == r
                kg[sel] = x[src[sel]] @ W[r]
            out.append((kg / cnt[typ][:, None], lin(x[partners]) / self.deg[t]))
        return out

    def _layer(self, k, x_t, msgs, m_kg, m_ddi):
        """Masked output of layer k at a target drug (eval mode: no dropout)."""
        conv, norm = self.model.convs[k], self.model.norms[k]
        kg, ddi = msgs[k]
        h = m_kg @ kg + x_t @ conv.root + conv.bias + m_ddi @ ddi
        return x_t + F.relu(norm(h))

    def _embed(self, t, msgs, m_kg, m_ddi):
        x1_t = self._layer(0, self.states[0][t], msgs, m_kg, m_ddi)
        # layer 2 combines neighbours' (fixed) layer-1 states with t's own masked layer-1 state
        return self._layer(1, x1_t, msgs, m_kg, m_ddi)

    def _logit(self, ha, hb):
        return self.model.ddi_head(self.model.sym(ha[None], hb[None])).squeeze()

    # ------------------------------------------------------------ main entry
    def explain(self, a, b, names, ntypes, steps=150, lr=0.05, size_coef=0.005, ent_coef=1.0, top=8, seed=0):
        """names/ntypes: callables node id -> display name / node type."""
        self.model.eval()
        parts = [self._incoming(a), self._incoming(b)]
        msgs = [self._messages(t, *p) for t, p in zip((a, b), parts)]
        sizes = [(len(s), len(p)) for s, _, p in parts]
        n_edges = sum(k + d for k, d in sizes)

        def split(m):
            out, i = [], 0
            for k, d in sizes:
                out.append((m[i:i + k], m[i + k:i + k + d]))
                i += k + d
            return out

        def logit(m):
            (ka, da), (kb, db) = split(m)
            ha = self._embed(a, msgs[0], ka, da)
            hb = self._embed(b, msgs[1], kb, db)
            return self._logit(ha, hb)

        with torch.no_grad():
            full = logit(torch.ones(n_edges))
            check = self.model.ddi_logit(self.states[-1][:self.D], torch.tensor([a]), torch.tensor([b]))[0]
            assert abs(float(full) - float(check)) < 1e-3, (float(full), float(check))
            empty = logit(torch.zeros(n_edges))
        target = float(full > 0)

        g = torch.Generator().manual_seed(seed + 7919 * a + b)
        param = (torch.randn(n_edges, generator=g) * 0.1).requires_grad_()
        opt = torch.optim.Adam([param], lr=lr)
        for _ in range(steps):
            m = torch.sigmoid(param)
            loss = F.binary_cross_entropy_with_logits(logit(m), torch.tensor(target))
            ent = -m * torch.log(m + 1e-15) - (1 - m) * torch.log(1 - m + 1e-15)
            loss = loss + size_coef * m.sum() + ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        mask = torch.sigmoid(param).detach()

        # edge list with display info
        edges = []
        for side, ((src, typ, partners), (km, dm)) in enumerate(zip(parts, split(mask))):
            drug = (a, b)[side]
            for s, t, w in zip(src.tolist(), typ.tolist(), km.tolist()):
                edges.append((w, s, drug, self.rel_names[t]))
            for p, w in zip(partners.tolist(), dm.tolist()):
                edges.append((w, p, drug, "ddi"))
        edges.sort(key=lambda e: -e[0])

        # node importance: sum of its edge masks (a node linked to both drugs can score up to 2+)
        nodes = {}
        for w, s, drug, rel in edges:
            if s in (a, b):
                continue
            n = nodes.setdefault(s, {"id": int(s), "name": names(s), "type": ntypes(s), "score": 0.0,
                                     "drugs": set(), "rels": set()})
            n["score"] += w
            if w >= 0.5:
                n["drugs"].add(int(drug))
                n["rels"].add(REL_LABEL.get(rel, rel))
        ranked = sorted(nodes.values(), key=lambda n: -n["score"])
        best = ranked[0]["score"] if ranked else 1.0
        top_nodes = []
        for n in ranked[:top]:
            top_nodes.append({**n, "score": round(n["score"] / best, 3),
                              "drugs": sorted(n["drugs"]), "rels": sorted(n["rels"]),
                              "shared": len(n["drugs"]) == 2})

        # fidelity: switch off the top nodes' edges and see how far the score moves
        top_ids = {n["id"] for n in top_nodes}
        keep = []
        for (src, _, partners) in parts:
            keep += [0.0 if s in top_ids else 1.0 for s in src.tolist()]
            keep += [0.0 if p in top_ids else 1.0 for p in partners.tolist()]
        with torch.no_grad():
            without = logit(torch.tensor(keep))
            only = logit(torch.tensor(keep).neg().add(1.0))

        def p(x):
            return round(float(torch.sigmoid(x)), 4)
        return {
            "method": "GNNExplainer (edge mask over the edges feeding both drugs, R-GCN)",
            "score": p(full), "score_without_top": p(without), "score_only_top": p(only), "score_no_edges": p(empty),
            "n_edges": int(n_edges), "n_selected": int((mask >= 0.5).sum()),
            "top_nodes": top_nodes,
            "top_edges": [{"source": int(s), "source_name": names(s), "drug": int(d), "rel": REL_LABEL.get(r, r),
                           "weight": round(float(w), 3)} for w, s, d, r in edges[:top]],
        }
