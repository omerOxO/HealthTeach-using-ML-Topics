"""Path-based explanations grounded in real knowledge-graph edges.

For a drug pair (A, B) we enumerate
- 2-hop meta-paths through a shared node: A -enzyme- CYP2C9 -enzyme- B, A -target- PTGS1 -target- B,
  shared transporter/carrier, shared single-drug side effect (SIDER), shared ATC class;
- 3-hop protein paths through a protein-protein interaction: A -target- P1 -ppi- P2 -target- B;
- an "effect tail": a single-drug side effect of A or B that matches a predicted/TWOSIDES pair effect.

Paths are scored by relation weight x node specificity (1 / log2(2 + number of drugs touching the
node)), so a shared CYP3A4 (hundreds of drugs) ranks below a rarely shared target.
"""
import math
import re

REL_WEIGHT = {"enzyme": 1.0, "target": 0.95, "transporter": 0.85, "carrier": 0.6,
              "side_effect": 0.7, "in_class": 0.55, "contraindication": 0.3, "indication": 0.25,
              "off_label": 0.2}
MECHANISM_RELS = {"enzyme", "target", "transporter", "carrier"}

SENTENCE = {
    "enzyme": "{A} and {B} are both processed by the enzyme {N}. When two drugs share a metabolising "
              "enzyme, one can slow the breakdown of the other, so its level in the blood can rise.",
    "target": "{A} and {B} both act on {N}, so their effects on the body can add up.",
    "transporter": "{A} and {B} both rely on the transporter {N}, which moves drugs into and out of cells. "
                   "Competing for it can change how much of each drug reaches the blood.",
    "carrier": "{A} and {B} both bind the carrier protein {N} in the blood. One can push the other off, "
               "which raises the amount of free, active drug.",
    "side_effect": "{A} and {B} are each linked to {N} on their own, so taking them together may add to "
                   "that risk.",
    "in_class": "{A} and {B} belong to the same drug class ({N}), so their effects can overlap.",
    "contraindication": "{A} and {B} are both flagged as unsuitable in {N}.",
    "indication": "{A} and {B} are both used for {N}, so they may be doubling up on the same job.",
    "off_label": "{A} and {B} are both used off-label for {N}.",
}
PPI_SENTENCE = ("{A} acts on {P1}, which works together with {P2}, a protein that {B} acts on. "
                "This indirect link can let one drug change the other's effect.")
EFFECT_TAIL = "{X} on its own is linked to {E}, which is one reason {E} is flagged for this pair."

WORD = re.compile(r"[a-z]+")

# Interaction-type wording -> the side effects it implies (searched among SIDER single-drug effects)
MECHANISM_EFFECTS = [
    (r"anticoagulant|antiplatelet|bleeding|thrombolytic", ["hemorrhage", "bleeding", "haematoma", "ecchymosis"]),
    (r"qtc|arrhythmogenic|qt prolong", ["qt prolonged", "torsade de pointes", "arrhythmia"]),
    (r"serotonergic", ["serotonin syndrome"]),
    (r"cns depressant|sedative|respiratory depress", ["somnolence", "sedation", "respiratory depression"]),
    (r"orthostatic|hypotensive", ["hypotension", "orthostatic hypotension", "dizziness"]),
    (r"hypertensive|vasopressor|vasoconstrict", ["hypertension"]),
    (r"myopathic|rhabdomyolysis", ["rhabdomyolysis", "myopathy", "myalgia"]),
    (r"hypoglycemic", ["hypoglycaemia", "hypoglycemia"]),
    (r"hyperkalemi", ["hyperkalaemia", "hyperkalemia"]),
    (r"hypokalemic", ["hypokalaemia", "hypokalemia"]),
    (r"hyponatremic", ["hyponatraemia", "hyponatremia"]),
    (r"nephrotoxic", ["renal failure", "nephrotoxicity", "renal impairment"]),
    (r"hepatotoxic", ["hepatotoxicity", "hepatitis", "hepatic failure"]),
    (r"cardiotoxic|heart failure", ["cardiac failure", "heart failure"]),
    (r"bradycardic|av block|atrioventricular", ["bradycardia", "atrioventricular block"]),
    (r"tachycardic", ["tachycardia"]),
    (r"neuroexcitatory|stimulatory", ["convulsion", "seizure", "insomnia"]),
    (r"ulcerogenic", ["peptic ulcer", "gastrointestinal hemorrhage"]),
    (r"anticholinergic", ["dry mouth", "constipation", "urinary retention"]),
    (r"myelosuppressive", ["neutropenia", "thrombocytopenia", "anaemia"]),
    (r"hypersensitivity", ["hypersensitivity", "rash"]),
]


PK_PATTERN = re.compile(r"serum concentration|metabolism|bioavailability|absorption|excretion|protein binding|"
                        r"active metabolites")


def mechanism_effects(sentence):
    if not sentence:
        return []
    s = sentence.lower()
    out = []
    for pattern, terms in MECHANISM_EFFECTS:
        if re.search(pattern, s):
            out += terms
    return out


def _norm_effect(s):
    s = s.lower().replace("haem", "hem").replace("oedema", "edema").replace("aemia", "emia")
    return set(WORD.findall(s)) - {"of", "the", "and", "nos", "disorder", "abnormal"}


def effects_match(a, b):
    ta, tb = _norm_effect(a), _norm_effect(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


class PathExplainer:
    def __init__(self, G, n_drugs):
        self.G = G
        self.n_drugs = n_drugs
        self.drug_deg = {}
        for n in G.nodes:
            if n >= n_drugs:
                self.drug_deg[n] = sum(1 for m in G.neighbors(n) if m < n_drugs)

    # ---- helpers
    def name(self, n):
        return self.G.nodes[n]["name"]

    def ntype(self, n):
        return self.G.nodes[n]["ntype"]

    def rels(self, u, v):
        return {d["rel"] for d in self.G.get_edge_data(u, v).values()}

    def spec(self, n):
        return 1.0 / math.log2(2 + self.drug_deg.get(n, 0))

    def neighbours(self, d, rels=None):
        out = {}
        for n, edata in self.G[d].items():
            for e in edata.values():
                if rels is None or e["rel"] in rels:
                    out.setdefault(n, set()).add(e["rel"])
        return out

    # ---- main entry
    def explain(self, a, b, pair_effects=(), mechanism=None, max_paths=6):
        """pair_effects: predicted or TWOSIDES-reported side effects for the pair.
        mechanism: the (known or predicted) DrugBank interaction sentence; it implies which effects
        matter most (an anticoagulant interaction -> bleeding), see MECHANISM_EFFECTS."""
        A, B = self.name(a), self.name(b)
        na, nb = self.neighbours(a), self.neighbours(b)
        focus = mechanism_effects(mechanism)
        mech = (mechanism or "").lower()
        pk = bool(PK_PATTERN.search(mech))  # pharmacokinetic: shared enzymes/transporters matter most
        pd_ = "activities" in mech or "risk or severity" in mech  # pharmacodynamic: shared targets/effects
        paths = []
        for n in set(na) & set(nb):
            for rel in na[n] & nb[n]:
                if rel == "ppi":
                    continue
                if rel == "in_class" and "combination" in self.name(n).lower():
                    continue  # ATC combination-product classes do not mean the drugs share a class
                w = REL_WEIGHT.get(rel, 0.2) * self.spec(n)
                if pk and rel in ("enzyme", "transporter", "carrier"):
                    w *= 3.0
                if pd_ and rel == "target":
                    w *= 1.5
                matched = False
                if rel == "side_effect":
                    if any(effects_match(self.name(n), e) for e in focus):
                        w, matched = w * 4.0, True
                    elif any(effects_match(self.name(n), e) for e in pair_effects):
                        w, matched = w * 1.6, True
                    else:
                        w *= 0.5  # a shared but unrelated side effect is weak evidence
                paths.append({
                    "kind": rel, "score": w, "via": [n], "matches_effect": matched,
                    "text": SENTENCE.get(rel, "{A} and {B} share {N}.").format(A=A, B=B, N=self.name(n)),
                    "edges": [(a, n, rel), (b, n, rel)],
                })

        # protein-protein bridges: indirect, so a low-weight fallback
        pa = {n for n, r in na.items() if r & MECHANISM_RELS}
        pb = {n for n, r in nb.items() if r & MECHANISM_RELS}
        bridges = []
        for p1 in pa - pb:
            for p2 in self.G.neighbors(p1):
                if p2 in pb and p2 not in pa and "ppi" in self.rels(p1, p2):
                    r1 = sorted(na[p1] & MECHANISM_RELS)[0]
                    r2 = sorted(nb[p2] & MECHANISM_RELS)[0]
                    w = 0.12 * min(REL_WEIGHT[r1], REL_WEIGHT[r2]) * math.sqrt(self.spec(p1) * self.spec(p2))
                    bridges.append({
                        "kind": "ppi", "score": w, "via": [p1, p2], "matches_effect": False,
                        "text": PPI_SENTENCE.format(A=A, B=B, P1=self.name(p1), P2=self.name(p2)),
                        "edges": [(a, p1, r1), (p1, p2, "ppi"), (b, p2, r2)],
                    })
        bridges.sort(key=lambda p: -p["score"])
        paths += bridges[:3]
        paths.sort(key=lambda p: -p["score"])

        # effect tail: a single-drug side effect of A or B matching the mechanism's effect first,
        # then a predicted/TWOSIDES pair effect. Skip effects the top path already names.
        # several shared enzymes: name the others too, since specificity favours the rarer one,
        # which is not necessarily the main metabolic pathway
        if paths and paths[0]["kind"] in ("enzyme", "transporter"):
            kind = paths[0]["kind"]
            others = sorted((self.name(n) for n in set(na) & set(nb)
                             if kind in na[n] and kind in nb[n] and n != paths[0]["via"][0]))
            if others:
                listed = ", ".join(others[:4]) + (" and others" if len(others) > 4 else "")
                paths[0]["text"] += f" They also share {listed}."

        shown = {v for p in paths[:1] for v in p["via"]}
        tail = None
        for terms in (focus,):
            for e in terms:
                for d, nd in ((b, nb), (a, na)):
                    hit = next((n for n, r in nd.items() if "side_effect" in r and n not in shown
                                and effects_match(self.name(n), e)), None)
                    if hit is not None:
                        eff = self.name(hit)
                        tail = {"drug": d, "node": hit, "effect": eff,
                                "text": EFFECT_TAIL.format(X=self.name(d), E=eff.lower())}
                        break
                if tail:
                    break
            if tail:
                break

        top = paths[:max_paths]
        return {"paths": [self._public(p) for p in top], "effect_tail": tail and {
            "drug": tail["drug"], "node": tail["node"], "effect": tail["effect"], "text": tail["text"]},
            "graph": self._graph(a, b, top, tail)}

    def _public(self, p):
        return {"kind": p["kind"], "score": round(p["score"], 4), "text": p["text"],
                "via": [{"id": int(n), "name": self.name(n), "type": self.ntype(n)} for n in p["via"]],
                "matches_effect": p["matches_effect"]}

    def _graph(self, a, b, paths, tail):
        nodes, edges = {}, []

        def add(n):
            nodes[n] = {"id": int(n), "label": self.name(n), "type": self.ntype(n),
                        "focus": n in (a, b)}
        add(a)
        add(b)
        for k, p in enumerate(paths[:4]):
            for u, v, rel in p["edges"]:
                add(u)
                add(v)
                edges.append({"source": int(u), "target": int(v), "rel": rel, "hl": k == 0})
        if tail:
            add(tail["node"])
            edges.append({"source": int(tail["drug"]), "target": int(tail["node"]), "rel": "side_effect", "hl": True})
        seen, uniq = set(), []
        for e in edges:
            k = (e["source"], e["target"], e["rel"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return {"nodes": list(nodes.values()), "edges": uniq}

    def neighbourhood(self, d, per_type=12):
        """Neighbours of a drug for the graph explorer, most specific first, capped per node type."""
        groups = {}
        for n, rels in self.neighbours(d).items():
            groups.setdefault(self.ntype(n), []).append((self.spec(n), n, sorted(rels)))
        nodes = [{"id": int(d), "label": self.name(d), "type": "drug", "focus": True}]
        edges = []
        for t, items in groups.items():
            items.sort(key=lambda x: -x[0])
            for _, n, rels in items[:per_type]:
                nodes.append({"id": int(n), "label": self.name(n), "type": t, "focus": False})
                for r in rels:
                    edges.append({"source": int(d), "target": int(n), "rel": r})
        counts = {t: len(v) for t, v in groups.items()}
        return {"nodes": nodes, "edges": edges, "counts": counts}
