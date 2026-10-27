"""Graph model: R-GCN or relational GAT over the heterogeneous KG + pair decoders.

    python -m ddi.gnn --labels merged --split random --conv rgcn
    python -m ddi.gnn --labels merged --split all --conv all
    python -m ddi.gnn --labels merged --split all --conv rgcn --drug-encoder gin

Encoder
- drug nodes start from their Morgan fingerprint (--drug-encoder fp, the default) or from a GIN over
  the drug's molecular graph (--drug-encoder gin: RDKit atoms and bonds, 3 GINEConv layers, mean+max
  pooling, trained end to end), so unseen drugs still get an embedding;
  proteins, side effects, diseases and ATC classes start from learned embeddings
- each layer: relational conv (RGCNConv, or one GATConv per relation for R-GAT) over the biological relations
  (target/enzyme/transporter/carrier, side_effect, indication, contraindication, off_label,
  in_class, ppi, and their reverses) plus a DDI channel: mean over known interaction partners,
  computed as a dense 1.7k x 1.7k matmul
- training-time target masking: the positives supervised in a batch are removed from the DDI
  channel for that step, so the model cannot read the answer off the graph

Decoders (multi-task)
- interaction:    MLP([h_a*h_b, |h_a-h_b|]) -> P(interaction), symmetric
- type:           MLP([h_a, h_b, h_a*h_b]) -> 86 DrugBank interaction types (ordered: a = #Drug1)
- side effects:   MLP([h_a*h_b, |h_a-h_b|]) -> TWOSIDES side effects (multi-label, Decagon-style)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GINEConv, RGCNConv, global_max_pool, global_mean_pool

from .config import ARTIFACTS, PROCESSED, SEED
from .metrics import evaluate_split, per_label_auprc, print_result, save_result
from .splits import LABEL_SETS, TrainNegatives, load_positive_pairs, load_split

SYMMETRIC_RELS = {"ppi"}


# ---------------------------------------------------------------- graph tensors
def load_graph():
    nodes = pd.read_parquet(PROCESSED / "nodes.parquet")
    edges = pd.read_parquet(PROCESSED / "edges.parquet")
    fps = np.load(PROCESSED / "fingerprints.npy")
    rels = sorted(edges["rel"].unique())
    rel_names, src, dst, typ = [], [], [], []
    for r in rels:
        e = edges[edges["rel"] == r]
        fid = len(rel_names)
        rel_names.append(r)
        src.append(e["src"].to_numpy()); dst.append(e["dst"].to_numpy()); typ.append(np.full(len(e), fid))
        if r in SYMMETRIC_RELS:
            rid = fid
        else:
            rid = len(rel_names)
            rel_names.append(f"rev_{r}")
        src.append(e["dst"].to_numpy()); dst.append(e["src"].to_numpy()); typ.append(np.full(len(e), rid))
    edge_index = torch.from_numpy(np.stack([np.concatenate(src), np.concatenate(dst)])).long()
    edge_type = torch.from_numpy(np.concatenate(typ)).long()
    n_drugs = int((nodes["ntype"] == "drug").sum())
    assert (nodes["ntype"].iloc[:n_drugs] == "drug").all(), "drug nodes must come first"
    return {
        "n_nodes": len(nodes), "n_drugs": n_drugs, "rel_names": rel_names,
        "edge_index": edge_index, "edge_type": edge_type, "fp": torch.from_numpy(fps).float(),
    }


# ---------------------------------------------------------------- molecular graphs (GIN drug encoder)
ATOMS = ["C", "N", "O", "S", "F", "Cl", "P", "I", "Br", "B", "Si", "Se", "Na", "K", "Li", "Ca", "Mg", "Al",
         "Fe", "Zn", "Cu", "Co", "Pt", "Au", "Hg", "As", "Sb", "Bi", "Ti"]
DEGREES, CHARGES, NUM_HS = [0, 1, 2, 3, 4, 5], [-1, 0, 1], [0, 1, 2, 3]
HYBRID = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
BONDS = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_DIM = sum(len(c) + 1 for c in (ATOMS, DEGREES, CHARGES, NUM_HS, HYBRID)) + 2
BOND_DIM = len(BONDS) + 2


def _onehot(v, choices):
    out = [0.0] * (len(choices) + 1)  # last slot = other
    out[choices.index(v) if v in choices else -1] = 1.0
    return out


def _mol_data(smiles):
    from .chem import standard_mol
    mol = standard_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0:  # one blank atom, so every drug has a graph
        return Data(x=torch.zeros(1, ATOM_DIM), edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, BOND_DIM))
    x = [_onehot(a.GetSymbol(), ATOMS) + _onehot(a.GetDegree(), DEGREES) + _onehot(a.GetFormalCharge(), CHARGES)
         + _onehot(a.GetTotalNumHs(), NUM_HS) + _onehot(str(a.GetHybridization()), HYBRID)
         + [float(a.GetIsAromatic()), float(a.IsInRing())] for a in mol.GetAtoms()]
    src, dst, attr = [], [], []
    for bd in mol.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        f = _onehot(str(bd.GetBondType()), BONDS)[:-1] + [float(bd.GetIsConjugated()), float(bd.IsInRing())]
        src += [i, j]
        dst += [j, i]
        attr += [f, f]
    return Data(x=torch.tensor(x), edge_index=torch.tensor([src, dst], dtype=torch.long).view(2, -1),
                edge_attr=torch.tensor(attr).view(-1, BOND_DIM))


def load_mol_batch():
    """All drug molecules as one PyG Batch in drug-index order (cached in data/processed/mol_graphs.pt)."""
    cache = PROCESSED / "mol_graphs.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    drugs = pd.read_parquet(PROCESSED / "drugs.parquet").sort_values("drug_idx")
    batch = Batch.from_data_list([_mol_data(s) for s in drugs["smiles"]])
    torch.save(batch, cache)
    return batch


class MolGIN(nn.Module):
    """GIN over each drug's molecular graph (GINEConv, so bond features enter the messages) -> drug vector."""

    def __init__(self, hidden, layers=3, dropout=0.2):
        super().__init__()
        self.atom_in = nn.Linear(ATOM_DIM, hidden)
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                           nn.Linear(hidden, hidden)), edge_dim=BOND_DIM)
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.out = nn.Linear(2 * hidden, hidden)
        self.dropout = dropout

    def forward(self, mol):
        x = self.atom_in(mol.x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.dropout(F.relu(norm(conv(x, mol.edge_index, mol.edge_attr))), self.dropout, self.training)
        return self.out(torch.cat([global_mean_pool(x, mol.batch), global_max_pool(x, mol.batch)], -1))


def drug_inputs(g, drug_encoder):
    """The drug input DDIModel.encode takes: the fingerprint matrix or the batch of molecules."""
    return load_mol_batch() if drug_encoder == "gin" else g["fp"]


def model_name(conv, drug_encoder="fp"):
    return f"gnn_{conv}" + ("_gin" if drug_encoder == "gin" else "")


def ddi_adjacency(n_drugs, a, b, device):
    A = torch.zeros(n_drugs, n_drugs, device=device)
    a, b = torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
    A[a, b] = 1.0
    A[b, a] = 1.0
    return A


def row_normalise(A):
    return A / A.sum(1, keepdim=True).clamp(min=1.0)


# ---------------------------------------------------------------- model
class RelGATConv(nn.Module):
    """Relational GAT: one GATConv per relation, summed, plus a root transform.

    PyG's RGATConv materialises a weight matrix per edge (E x in x heads*out), which needs ~30 GB
    on this graph; per-relation GATConv only stores per-edge attention scores."""

    def __init__(self, hidden, n_rel, heads=2):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(hidden, hidden, heads=heads, concat=False, add_self_loops=False)
                                    for _ in range(n_rel)])
        self.root = nn.Linear(hidden, hidden)
        self._masks = None

    def forward(self, x, edge_index, edge_type):
        if self._masks is None or self._masks[0] is not edge_type:
            self._masks = (edge_type, [edge_index[:, edge_type == r] for r in range(len(self.convs))])
        out = self.root(x)
        for conv, ei in zip(self.convs, self._masks[1]):
            if ei.numel():
                out = out + conv(x, ei)
        return out


def mlp(i, h, o, dropout):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


class DDIModel(nn.Module):
    def __init__(self, n_nodes, n_drugs, fp_dim, n_rel, n_types, n_se, hidden=128, layers=2,
                 conv="rgcn", dropout=0.2, num_bases=8, heads=2, drug_encoder="fp"):
        super().__init__()
        self.n_drugs, self.conv_kind = n_drugs, conv
        self.drug_in = MolGIN(hidden, dropout=dropout) if drug_encoder == "gin" else mlp(fp_dim, hidden, hidden, dropout)
        self.node_emb = nn.Embedding(n_nodes - n_drugs, hidden)
        nn.init.normal_(self.node_emb.weight, std=0.1)
        self.convs, self.ddi_lin, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            if conv == "rgcn":
                self.convs.append(RGCNConv(hidden, hidden, n_rel, num_bases=num_bases))
            else:
                self.convs.append(RelGATConv(hidden, n_rel, heads=heads))
            self.ddi_lin.append(nn.Linear(hidden, hidden, bias=False))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout
        self.ddi_head = mlp(2 * hidden, hidden, 1, dropout)
        self.type_head = mlp(3 * hidden, hidden, n_types, dropout)
        self.se_head = mlp(2 * hidden, hidden, n_se, dropout) if n_se else None

    def encode(self, drug_x, edge_index, edge_type, A_norm=None):
        """drug_x: fingerprint matrix (fp encoder) or molecule Batch (gin encoder)."""
        x = torch.cat([self.drug_in(drug_x), self.node_emb.weight], 0)
        D = self.n_drugs
        for conv, lin, norm in zip(self.convs, self.ddi_lin, self.norms):
            h = conv(x, edge_index, edge_type)
            if A_norm is not None:
                h = torch.cat([h[:D] + lin(A_norm @ x[:D]), h[D:]], 0)
            h = F.dropout(F.relu(norm(h)), self.dropout, self.training)
            x = x + h
        return x[:D]

    @staticmethod
    def sym(ha, hb):
        return torch.cat([ha * hb, (ha - hb).abs()], -1)

    def ddi_logit(self, h, a, b):
        return self.ddi_head(self.sym(h[a], h[b])).squeeze(-1)

    def type_logits(self, h, a, b):
        return self.type_head(torch.cat([h[a], h[b], h[a] * h[b]], -1))

    def se_logits(self, h, a, b):
        return self.se_head(self.sym(h[a], h[b]))


# ---------------------------------------------------------------- data for the extra heads
def twosides_tensors(n_drugs):
    tw = pd.read_parquet(PROCESSED / "twosides.parquet")
    n_se = len(pd.read_parquet(PROCESSED / "side_effects.parquet"))
    pairs = tw[["a", "b"]].drop_duplicates().reset_index(drop=True)
    pid = {(a, b): i for i, (a, b) in enumerate(pairs.itertuples(index=False))}
    Y = np.zeros((len(pairs), n_se), dtype=np.uint8)
    Y[[pid[(a, b)] for a, b in zip(tw["a"], tw["b"])], tw["se_idx"].to_numpy()] = 1
    return pairs.to_numpy(), Y, n_se


def twosides_split(tw_pairs, split, kind, seed=SEED):
    if kind == "random":
        perm = np.random.default_rng(seed + 1).permutation(len(tw_pairs))
        n = len(perm) // 10
        return {"val": perm[:n], "test": perm[n:2 * n], "train": perm[2 * n:]}
    role = np.zeros(len(split["degree"]), dtype=int)
    role[split["val_drugs"]], role[split["test_drugs"]] = 1, 2
    ra, rb = role[tw_pairs[:, 0]], role[tw_pairs[:, 1]]
    return {"train": np.flatnonzero((ra == 0) & (rb == 0)),
            "val": np.flatnonzero(((ra == 1) | (rb == 1)) & (ra != 2) & (rb != 2)),
            "test": np.flatnonzero((ra == 2) | (rb == 2))}


# ---------------------------------------------------------------- training
def run(labels, kind, conv, epochs=60, batch=8192, se_batch=2048, lr=2e-3, hidden=128, patience=8,
        device=None, drug_encoder="fp"):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = load_graph()
    D = g["n_drugs"]
    ddi = load_positive_pairs(labels)
    pairs = ddi[["a", "b"]].to_numpy()
    split = load_split(labels, kind)
    types = pd.read_parquet(PROCESSED / "ddi_types.parquet")
    type_index = {t: i for i, t in enumerate(types["type_id"])}

    tw_pairs, tw_Y, n_se = twosides_tensors(D)
    tw_split = twosides_split(tw_pairs, split, kind)

    model = DDIModel(g["n_nodes"], D, g["fp"].shape[1], len(g["rel_names"]), len(types), n_se,
                     hidden=hidden, conv=conv, drug_encoder=drug_encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    fp, ei, et = drug_inputs(g, drug_encoder).to(device), g["edge_index"].to(device), g["edge_type"].to(device)
    name = model_name(conv, drug_encoder)

    tr = split["train_pos"]
    A_train = ddi_adjacency(D, pairs[tr, 0], pairs[tr, 1], device)
    A_train_norm = row_normalise(A_train)
    negs = TrainNegatives(ddi, split, D)

    # typed training pairs (ordered: drug1 -> drug2)
    typed = ddi.iloc[tr]
    typed = typed[typed["tdc_type"] >= 0]
    t_d1 = typed["tdc_drug1"].to_numpy()
    t_d2 = np.where(typed["a"].to_numpy() == t_d1, typed["b"].to_numpy(), typed["a"].to_numpy())
    t_y = typed["tdc_type"].map(type_index).to_numpy()
    typed_pos = {k: i for i, k in enumerate(typed.index)}  # ddi row -> typed row

    @torch.no_grad()
    def embed():
        model.eval()
        return model.encode(fp, ei, et, A_train_norm)

    def scorer(h):
        @torch.no_grad()
        def f(a, b):
            out = []
            for i in range(0, len(a), 200_000):
                out.append((model.ddi_logit(h, torch.as_tensor(a[i:i + 200_000], device=device),
                                                          torch.as_tensor(b[i:i + 200_000], device=device))).cpu())
            return torch.cat(out).numpy()
        return f

    def val_auprc():
        f = scorer(embed())
        vp, vn = split["val_pos"], split["val_neg"]
        s = np.r_[f(pairs[vp, 0], pairs[vp, 1]), f(vn[:, 0], vn[:, 1])]
        return average_precision_score(np.r_[np.ones(len(vp)), np.zeros(len(vn))], s)

    best, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        tot = n = 0
        for i in range(0, len(order), batch):
            idx = tr[order[i:i + batch]]
            pa, pb = pairs[idx, 0], pairs[idx, 1]
            A = A_train.clone()
            ta, tb = torch.as_tensor(pa, device=device), torch.as_tensor(pb, device=device)
            A[ta, tb] = 0.0
            A[tb, ta] = 0.0
            h = model.encode(fp, ei, et, row_normalise(A))

            na, nb = negs(len(idx))
            na, nb = torch.as_tensor(na, device=device), torch.as_tensor(nb, device=device)
            pos_l, neg_l = model.ddi_logit(h, ta, tb), model.ddi_logit(h, na, nb)
            loss = F.binary_cross_entropy_with_logits(pos_l, torch.ones_like(pos_l)) + \
                F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l))

            rows = [typed_pos[k] for k in idx if k in typed_pos]
            if rows:
                d1 = torch.as_tensor(t_d1[rows], device=device)
                d2 = torch.as_tensor(t_d2[rows], device=device)
                loss = loss + 0.5 * F.cross_entropy(model.type_logits(h, d1, d2),
                                                    torch.as_tensor(t_y[rows], device=device))
            if n_se:
                j = rng.choice(tw_split["train"], min(se_batch, len(tw_split["train"])), replace=False)
                sa = torch.as_tensor(tw_pairs[j, 0], device=device)
                sb = torch.as_tensor(tw_pairs[j, 1], device=device)
                loss = loss + F.binary_cross_entropy_with_logits(
                    model.se_logits(h, sa, sb), torch.as_tensor(tw_Y[j], device=device).float())

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        v = val_auprc()
        print(f"[{name} {labels}/{kind}] epoch {ep:3d} loss={tot / n:.4f} val AUPRC={v:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    h = embed()

    res = evaluate_split(scorer(h), pairs, split)
    res["train_seconds"] = round(train_s, 1)
    res["epochs"] = ep

    # interaction type (test positives that have a DrugBank type)
    te = ddi.iloc[split["test_pos"]]
    te = te[te["tdc_type"] >= 0]
    if len(te):
        d1 = te["tdc_drug1"].to_numpy()
        d2 = np.where(te["a"].to_numpy() == d1, te["b"].to_numpy(), te["a"].to_numpy())
        with torch.no_grad():
            pred = model.type_logits(h, torch.as_tensor(d1, device=device),
                                     torch.as_tensor(d2, device=device)).argmax(-1).cpu().numpy()
        yt = te["tdc_type"].map(type_index).to_numpy()
        res["type"] = {"accuracy": float(accuracy_score(yt, pred)),
                       "macro_f1": float(f1_score(yt, pred, average="macro")), "n": int(len(te))}

    # polypharmacy side effects (TWOSIDES test pairs)
    if n_se and len(tw_split["test"]):
        j = tw_split["test"]
        with torch.no_grad():
            s = torch.sigmoid(model.se_logits(h, torch.as_tensor(tw_pairs[j, 0], device=device),
                                              torch.as_tensor(tw_pairs[j, 1], device=device))).cpu().numpy()
        se_res = per_label_auprc(tw_Y[j], s)
        res["side_effects"] = {k: v for k, v in se_res.items() if k != "per_label_auprc"}
        res["side_effects"]["n_pairs"] = int(len(j))
        (ARTIFACTS / f"se_auprc_{name[4:]}_{labels}_{kind}.json").write_text(json.dumps(se_res["per_label_auprc"]))

    save_result(name, labels, kind, res)
    print_result(f"{name} | labels={labels} | split={kind}", res)
    if "type" in res:
        print(f"  type      acc={res['type']['accuracy']:.4f} macroF1={res['type']['macro_f1']:.4f}")
    if "side_effects" in res:
        print(f"  side eff. mean per-SE AUPRC={res['side_effects']['mean_auprc']:.4f} "
              f"AUROC={res['side_effects']['mean_auroc']:.4f} ({res['side_effects']['n_labels']} SEs)")

    torch.save({"state_dict": model.state_dict(),
                "config": {"n_nodes": g["n_nodes"], "n_drugs": D, "fp_dim": g["fp"].shape[1],
                           "n_rel": len(g["rel_names"]), "n_types": len(types), "n_se": n_se,
                           "hidden": hidden, "conv": conv, "drug_encoder": drug_encoder},
                "labels": labels, "split": kind, "threshold": float(1 / (1 + np.exp(-res["test"]["threshold"])))},
               ARTIFACTS / f"{name}_{labels}_{kind}.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=LABEL_SETS, default="merged")
    ap.add_argument("--split", choices=["random", "cold", "all"], default="all")
    ap.add_argument("--conv", choices=["rgcn", "rgat", "all"], default="rgcn")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--drug-encoder", choices=["fp", "gin"], default="fp",
                    help="initial drug features: Morgan fingerprint MLP or GIN over the molecular graph")
    args = ap.parse_args()
    kinds = ["random", "cold"] if args.split == "all" else [args.split]
    convs = ["rgcn", "rgat"] if args.conv == "all" else [args.conv]
    for kind in kinds:
        for c in convs:
            run(args.labels, kind, c, epochs=args.epochs, hidden=args.hidden, batch=args.batch,
                drug_encoder=args.drug_encoder)


if __name__ == "__main__":
    main()
