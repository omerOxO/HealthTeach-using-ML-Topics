"""Safer-alternative suggestions for a flagged pair.

For each drug X of the pair, candidates are drugs that could plausibly do the same job:
  * same main drug class (the class the app displays, which filters out the ATC classes PrimeKG copies
    from combination products), or
  * a large overlap in KG indications (Jaccard >= 0.25 with at least two shared indications).
Each candidate replaces X in the medication list and is scored against every other drug on the list with
the same known-or-predicted logic as the main analysis (severity, then combined risk). A candidate is kept
only if its worst severity is no worse than X's and its combined risk is lower.

This is a hypothesis generator, not a substitution recommendation: it ignores dose, efficacy, indication
for this patient and contraindications other than those in the knowledge graph.
"""
from .config import SEVERITY_ORDER, SEVERITY_WEIGHT
from .severity import pair_severity

RANK = {s: i for i, s in enumerate(SEVERITY_ORDER)}
MIN_SHARED_IND = 2
MIN_JACCARD = 0.25


def _ok_group(group):
    g = str(group)
    return "approved" in g and "withdrawn" not in g


class Alternatives:
    def __init__(self, svc):
        self.s = svc
        G = svc.G
        self.indications = {}
        for i in range(svc.n):
            if i in G:
                self.indications[i] = {n for n, ed in G[i].items() if any(e["rel"] == "indication" for e in ed.values())}
        self.by_class = {}
        for i in range(svc.n):
            c = svc.drug_card(i)["class"]
            if c:
                self.by_class.setdefault(c, []).append(i)
        self.approved = {i for i, g in enumerate(svc.drugs["group"]) if _ok_group(g)}

    def quick_pair(self, a, b):
        """Status, severity and probability for a pair, without the explanation work."""
        s = self.s
        a, b = min(a, b), max(a, b)
        k = s.known.get((a, b))
        p, (t_idx, _, _), _ = s._score(a, b)
        known = k is not None
        type_id = int(k.tdc_type) if known and k.tdc_type >= 0 else int(s.types["type_id"].iloc[t_idx])
        sev, _ = pair_severity(k.ddinter_level if known else None, type_id, known, p, s.threshold)
        status = "known" if known else ("predicted" if p >= s.threshold else "none")
        return {"status": status, "severity": sev, "score": round(p, 4)}

    def profile(self, x, others):
        rows = [{"with": self.s.drug_card(y), **self.quick_pair(x, y)} for y in others]
        prod = 1.0
        for r in rows:
            if r["severity"] != "none":
                prod *= 1 - SEVERITY_WEIGHT[r["severity"]] * (1.0 if r["status"] == "known" else r["score"])
        worst = min((RANK[r["severity"]] for r in rows), default=RANK["none"])
        return {"worst": SEVERITY_ORDER[worst], "risk": round(1 - prod, 4), "pairs": rows}

    def candidates(self, x):
        out = {}
        cls = self.s.drug_card(x)["class"]
        for c in self.by_class.get(cls, []) if cls else []:
            out[c] = f"same class: {cls}"
        ind_x = self.indications.get(x, set())
        if ind_x:
            for c, ind_c in self.indications.items():
                shared = ind_x & ind_c
                if c != x and len(shared) >= MIN_SHARED_IND and len(shared) / len(ind_x | ind_c) >= MIN_JACCARD:
                    names = sorted(self.s.G.nodes[n]["name"] for n in shared)[:3]
                    reason = "shares indications: " + ", ".join(names)
                    out[c] = out[c] + "; " + reason if c in out else reason
        out.pop(x, None)
        return {c: r for c, r in out.items() if c in self.approved}

    def suggest(self, a, b, med_list, per_drug=5):
        """med_list: drug indices of the whole regimen (the pair is added if missing)."""
        meds = list(dict.fromkeys([*med_list, a, b]))
        result = []
        for x in (a, b):
            others = [y for y in meds if y != x]
            current = self.profile(x, others)
            sugg = []
            cands = self.candidates(x)
            for c, reason in cands.items():
                if c in meds:
                    continue
                prof = self.profile(c, others)
                if RANK[prof["worst"]] < RANK[current["worst"]] or prof["risk"] >= current["risk"]:
                    continue
                sugg.append({"drug": self.s.drug_card(c), "reason": reason, **prof,
                             "risk_drop": round(current["risk"] - prof["risk"], 4)})
            sugg.sort(key=lambda r: (RANK[r["worst"]] * -1, r["risk"]))
            result.append({"replaces": self.s.drug_card(x), "current": current,
                           "n_candidates": len(cands), "suggestions": sugg[:per_drug]})
        return {"for": result, "medications": [self.s.drug_card(i) for i in meds],
                "note": ("Candidates share a drug class or several indications with the drug they replace and have a "
                         "lower known-or-predicted interaction risk with the rest of this list. They are not checked "
                         "for dose, efficacy or this patient's indication; discuss any switch with the prescriber.")}
