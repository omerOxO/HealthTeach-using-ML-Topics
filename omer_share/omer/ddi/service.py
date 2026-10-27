"""Inference service used by the FastAPI app: name lookup, pair scoring, severity, explanations."""
import itertools
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch

from .config import ARTIFACTS, DISCLAIMER, PROCESSED, REPORTS, SEVERITY_ORDER, SEVERITY_WEIGHT
from .explain import PathExplainer, effects_match, mechanism_effects
from .gnn import DDIModel, ddi_adjacency, drug_inputs, load_graph, row_normalise
from .model_explain import ModelExplainer, layer_states
from .ner import PrescriptionNER
from .rxnorm import RxNorm
from .severity import pair_severity

# Common English / brand names -> DrugBank name. Checked before RxNorm (ddi/rxnorm.py), so these work
# offline and need no network call; RxNorm covers everything else.
SYNONYMS = {
    "aspirin": "Acetylsalicylic acid", "asa": "Acetylsalicylic acid", "acetaminophen": "Acetaminophen",
    "paracetamol": "Acetaminophen", "tylenol": "Acetaminophen", "coumadin": "Warfarin",
    "jantoven": "Warfarin", "prilosec": "Omeprazole", "losec": "Omeprazole", "zocor": "Simvastatin",
    "lipitor": "Atorvastatin", "crestor": "Rosuvastatin", "advil": "Ibuprofen", "motrin": "Ibuprofen",
    "aleve": "Naproxen", "plavix": "Clopidogrel", "eliquis": "Apixaban", "xarelto": "Rivaroxaban",
    "glucophage": "Metformin", "zoloft": "Sertraline", "prozac": "Fluoxetine", "ultram": "Tramadol",
    "biaxin": "Clarithromycin", "lasix": "Furosemide", "norvasc": "Amlodipine", "synthroid": "Levothyroxine",
    "nexium": "Esomeprazole", "zithromax": "Azithromycin", "cipro": "Ciprofloxacin",
    "vitamin c": "Ascorbic acid", "vitamin d": "Cholecalciferol", "vitamin d2": "Ergocalciferol",
    "vitamin d3": "Cholecalciferol", "vitamin b12": "Cyanocobalamin", "vitamin k": "Phytonadione",
    "folate": "Folic acid", "vitamin e": "Vitamin E", "amoxil": "Amoxicillin", "claritin": "Loratadine",
    "zyrtec": "Cetirizine", "pepcid": "Famotidine", "protonix": "Pantoprazole", "singulair": "Montelukast",
    "cozaar": "Losartan", "zestril": "Lisinopril", "prinivil": "Lisinopril", "lanoxin": "Digoxin",
    "celexa": "Citalopram", "lexapro": "Escitalopram", "paxil": "Paroxetine", "xanax": "Alprazolam",
    "valium": "Diazepam", "neurontin": "Gabapentin", "lyrica": "Pregabalin", "toprol": "Metoprolol",
    "lopressor": "Metoprolol", "tenormin": "Atenolol", "deltasone": "Prednisone", "diflucan": "Fluconazole",
}

# Patient-context rules: (context flag, drug matcher on name/category/ATC text, message)
CONTEXT_RULES = [
    ("kidney", r"anti-inflammatory agents, non-steroidal|non-steroids|nsaid",
     "NSAIDs can worsen kidney function; avoid or monitor closely in kidney disease."),
    ("kidney", r"\bmetformin\b|aminoglycoside|\blithium\b",
     "Cleared by the kidneys; dose adjustment or monitoring is usually needed in kidney disease."),
    ("liver", r"hmg-coa reductase|statin|\bacetaminophen\b|hepatotoxic",
     "Processed by or can stress the liver; use caution in liver disease."),
    ("elderly", r"anticoagulant|platelet aggregation inhibitor|antithrombotic",
     "Bleeding risk from blood thinners rises with age (65+)."),
    ("elderly", r"benzodiazepine|anticholinergic|hypnotics and sedatives",
     "On the Beers list of drugs to avoid or use carefully in people aged 65+ (falls, confusion)."),
]


SALT_RE = re.compile(r"\s+(sodium|potassium|calcium|magnesium|hydrochloride|hcl|sulfate|maleate|mesylate|besylate|"
                     r"tartrate|succinate|citrate|acetate|bromide|fumarate|phosphate)$")

STOP_WORDS = {"and", "of", "for", "excl", "other", "agents", "drugs", "use", "products", "preparations"}


def fill_template(template, d1, d2):
    return template.replace("#Drug1", d1).replace("#Drug2", d2)


class DDIService:
    def __init__(self, model_file=None):
        self.drugs = pd.read_parquet(PROCESSED / "drugs.parquet")
        self.n = len(self.drugs)
        self.names = self.drugs["name"].tolist()
        self.ids = self.drugs["drugbank_id"].tolist()
        self.id2idx = {d: i for i, d in enumerate(self.ids)}
        self.types = pd.read_parquet(PROCESSED / "ddi_types.parquet")
        ses = pd.read_parquet(PROCESSED / "side_effects.parquet")
        self.side_effects = ses["se_name"].tolist()
        ddi = pd.read_parquet(PROCESSED / "ddi.parquet")
        self.known = {(r.a, r.b): r for r in ddi.itertuples(index=False)}
        tw = pd.read_parquet(PROCESSED / "twosides.parquet")
        self.twosides = tw.groupby(["a", "b"])["se_idx"].apply(list).to_dict()
        self.se_prior = ses["n_pairs"].to_numpy() / max(1, len(self.twosides))
        self.partners = {}
        for a, b in zip(ddi["a"], ddi["b"]):
            self.partners.setdefault(a, []).append(b)
            self.partners.setdefault(b, []).append(a)
        with open(PROCESSED / "kg.pkl", "rb") as f:
            self.G = pickle.load(f)
        self.explainer = PathExplainer(self.G, self.n)
        self._class_cache = {}
        self._build_name_index()
        self.use_rxnorm = os.environ.get("DDI_OFFLINE") != "1"
        self.rxnorm = RxNorm()
        self._ner = None
        self._load_model(model_file)

    # ------------------------------------------------------------ model
    def _load_model(self, model_file):
        path = model_file or ARTIFACTS / "gnn_rgcn_merged_random.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.model_info = {"file": path.name, "labels": ck["labels"], "split": ck["split"], **cfg}
        self.threshold = float(ck["threshold"])
        enc = cfg.get("drug_encoder", "fp")
        model = DDIModel(cfg["n_nodes"], cfg["n_drugs"], cfg["fp_dim"], cfg["n_rel"], cfg["n_types"],
                         cfg["n_se"], hidden=cfg["hidden"], conv=cfg["conv"], drug_encoder=enc)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        g = load_graph()
        keys = np.array(list(self.known.keys()))
        A = ddi_adjacency(self.n, keys[:, 0], keys[:, 1], "cpu")
        states = layer_states(model, drug_inputs(g, enc), g["edge_index"], g["edge_type"], row_normalise(A))
        self.h = states[-1][:self.n]
        self.model = model
        # GNNExplainer-style attributions for the pair page (R-GCN only)
        self.model_explainer = ModelExplainer(model, states, g["edge_index"], g["edge_type"], g["rel_names"], A,
                                              self.n) if cfg["conv"] == "rgcn" else None
        self._model_expl_cache = {}

    def model_explanation(self, a, b):
        """Which neighbour nodes and edges the model relied on for this pair (cached per pair)."""
        if self.model_explainer is None:
            return None
        a, b = min(a, b), max(a, b)
        if (a, b) not in self._model_expl_cache:
            if len(self._model_expl_cache) > 2000:
                self._model_expl_cache.clear()
            self._model_expl_cache[(a, b)] = self.model_explainer.explain(
                a, b, lambda n: self.G.nodes[n]["name"], lambda n: self.G.nodes[n]["ntype"])
        return self._model_expl_cache[(a, b)]

    @torch.no_grad()
    def _score(self, a, b):
        ta, tb = torch.tensor([a]), torch.tensor([b])
        p = torch.sigmoid(self.model.ddi_logit(self.h, ta, tb)).item()
        t_ab = torch.softmax(self.model.type_logits(self.h, ta, tb), -1)[0]
        t_ba = torch.softmax(self.model.type_logits(self.h, tb, ta), -1)[0]
        if t_ab.max() >= t_ba.max():
            t_idx, t_conf, drug1 = int(t_ab.argmax()), float(t_ab.max()), a
        else:
            t_idx, t_conf, drug1 = int(t_ba.argmax()), float(t_ba.max()), b
        se = torch.sigmoid(self.model.se_logits(self.h, ta, tb))[0].numpy()
        return p, (t_idx, t_conf, drug1), se

    # ------------------------------------------------------------ names
    def _build_name_index(self):
        self.lookup = {}
        for i, n in enumerate(self.names):
            self.lookup.setdefault(n.lower(), i)
        for i, d in enumerate(self.ids):
            self.lookup.setdefault(d.lower(), i)
        for syn, target in SYNONYMS.items():
            if target.lower() in self.lookup:
                self.lookup.setdefault(syn, self.lookup[target.lower()])

    def resolve(self, text):
        hits = self.resolve_all(text)["idxs"]
        return hits[0] if hits else None

    def resolve_all(self, text):
        """Local index (names, DrugBank ids, SYNONYMS) first; otherwise RxNorm brand/salt -> ingredient(s).
        A combination brand can map to several drugs. -> {"idxs", "source", "rxnorm"}"""
        key = str(text).strip().lower()
        i = self.lookup.get(key)
        if i is not None:
            return {"idxs": [i], "source": "local", "rxnorm": None}
        if not self.use_rxnorm or not re.search(r"[a-z]{3}", key) or key.startswith("db"):
            return {"idxs": [], "source": None, "rxnorm": None}
        rx = self.rxnorm.lookup(key)
        idxs = []
        ings = (rx or {}).get("ingredients", [])
        for ing in [c for c in ings if c["tty"] == "IN"] or [c for c in ings if c["tty"] == "PIN"]:
            base = ing["name"].lower()
            # the ingredient itself, its salt forms listed by RxNorm ("fluticasone propionate"), the salt stripped
            names = [base] + [c["name"].lower() for c in ings if c["tty"] == "PIN" and base in c["name"].lower()]
            for name in names + [SALT_RE.sub("", base)]:
                j = self.lookup.get(name)
                if j is not None:
                    if j not in idxs:
                        idxs.append(j)
                    break
        return {"idxs": idxs, "source": "rxnorm" if idxs else None, "rxnorm": rx}

    def search(self, q, limit=10):
        q = q.strip().lower()
        if not q:
            return []
        hits = []
        for key, i in self.lookup.items():
            if key.startswith(q):
                hits.append((0, len(key), key, i))
            elif q in key:
                hits.append((1, len(key), key, i))
        hits.sort()
        out, seen = [], set()
        for _, _, key, i in hits:
            if i in seen:
                continue
            seen.add(i)
            alias = key if key != self.names[i].lower() and not key.startswith("db") else None
            out.append({**self.drug_card(i), "matched": alias})
            if len(out) >= limit:
                break
        if len(q) >= 4 and not any(rank == 0 for rank, *_ in hits):
            # no local name starts with q: maybe a brand or salt name, ask RxNorm (exact names only, since
            # prefixes typed so far give fuzzy noise) and put its answer first
            r = self.resolve_all(q)
            rx = [{**self.drug_card(i), "matched": f"{q} (RxNorm)"}
                  for i in (r["idxs"] if (r["rxnorm"] or {}).get("match") == "exact" else [])]
            out = (rx + [o for o in out if o["idx"] not in {x["idx"] for x in rx}])[:limit]
        return out

    def extract(self, text):
        """Medication mentions in free text with dose / frequency / route (biomedical NER + name index,
        see ddi/ner.py). -> {"drugs", "mentions", "unmatched", "models"}"""
        if self._ner is None:
            self._ner = PrescriptionNER(self)
        return self._ner.extract(text)

    def _display_class(self, r):
        """PrimeKG lists every ATC class of every product containing the drug (simvastatin gets
        'DPP-4 inhibitors' from sitagliptin/simvastatin). Pick the class that is repeated most
        across the drug's own pharmacological categories."""
        classes = [c for v in (r["atc4"], r["atc3"]) if isinstance(v, str) for c in v.split("; ")
                   if "combination" not in c]
        if not classes:
            return None
        def stems(text):
            words = re.findall(r"[a-z]+", text.lower().replace("-", ""))
            return {w[:6] for w in words if w not in STOP_WORDS}
        entries = [stems(e) for e in str(r["categories"]).split("; ")]
        def score(c):  # how many category entries contain the whole class name
            toks = stems(c)
            hits = sum(1 for e in entries if toks and toks <= e)
            return hits * min(len(toks), 3)  # favour specific names over fragments like "derivatives"
        score = [score(c) for c in classes]  # ties keep ATC level-4 order
        return classes[int(np.argmax(score))]

    def drug_card(self, i):
        r = self.drugs.iloc[i]
        if i not in self._class_cache:
            self._class_cache[i] = self._display_class(r)
        return {"idx": int(i), "id": r["drugbank_id"], "name": r["name"], "class": self._class_cache[i]}

    # ------------------------------------------------------------ pair analysis
    def _pair(self, a, b, with_detail=False):
        a, b = min(a, b), max(a, b)
        A, B = self.names[a], self.names[b]
        k = self.known.get((a, b))
        p, (t_idx, t_conf, t_drug1), se = self._score(a, b)
        known = k is not None
        if known and k.tdc_type >= 0:
            type_id = int(k.tdc_type)
            d1 = int(k.tdc_drug1)
            tmpl = self.types.loc[self.types["type_id"] == type_id, "template"].iloc[0]
            mechanism, mech_source = fill_template(tmpl, self.names[d1], self.names[b if d1 == a else a]), "DrugBank"
        else:
            type_id = int(self.types["type_id"].iloc[t_idx])
            tmpl = self.types["template"].iloc[t_idx]
            other = b if t_drug1 == a else a
            mechanism = fill_template(tmpl, self.names[t_drug1], self.names[other])
            mech_source = "predicted"
        severity, sev_source = pair_severity(k.ddinter_level if known else None, type_id, known, p, self.threshold)
        tw_known = set(self.twosides.get((a, b), []))
        status = "known" if known else ("predicted" if p >= self.threshold else "none")
        ranked = self._rank_effects(se, mechanism if status != "none" else None)
        effects = [self._effect(j, se, tw_known) for j in ranked[:5]]
        row = {
            "a": self.drug_card(a), "b": self.drug_card(b), "status": status,
            "score": round(p, 4), "severity": severity, "severity_source": sev_source,
            "mechanism": mechanism if status != "none" else None, "mechanism_source": mech_source,
            "type_confidence": round(t_conf, 3) if mech_source == "predicted" else None,
            "effects": effects,
            "sources": {"drugbank_tdc": bool(known and k.in_tdc), "primekg": bool(known and k.in_primekg),
                        "ddinter": bool(known and k.in_ddinter),
                        "ddinter_level": k.ddinter_level if known and isinstance(k.ddinter_level, str) else None},
        }
        pair_effect_names = [e["name"] for e in effects[:3]]
        ex = self.explainer.explain(a, b, pair_effect_names, mechanism=mechanism if status != "none" else None)
        row["explanation"] = self._sentence(row, ex)
        row["path_chain"] = self._chain(a, b, ex)
        if with_detail:
            row["paths"] = ex["paths"]
            row["effect_tail"] = ex["effect_tail"]
            row["graph"] = ex["graph"]
            row["all_effects"] = [self._effect(j, se, tw_known) for j in ranked[:15]]
            row["twosides_count"] = len(tw_known)
            row["twosides_examples"] = [self.side_effects[j] for j in sorted(tw_known)[:12]]
            row["threshold"] = self.threshold
        return row

    def _rank_effects(self, se, mechanism=None):
        """Rank side effects by excess probability over their TWOSIDES base rate, so effects that are
        common for every pair do not top the list. Effects implied by the interaction mechanism
        (bleeding for an anticoagulant interaction) come first when the model rates them above base rate."""
        excess = se - self.se_prior
        focus = mechanism_effects(mechanism)
        promoted = np.array([bool(focus) and excess[j] > 0 and any(effects_match(n, t) for t in focus)
                             for j, n in enumerate(self.side_effects)])
        return np.lexsort((-excess, ~promoted))

    def _effect(self, j, se, tw_known):
        return {"name": self.side_effects[j], "prob": round(float(se[j]), 4),
                "base_rate": round(float(self.se_prior[j]), 4),
                "lift": round(float(se[j] / self.se_prior[j]), 2), "twosides": bool(j in tw_known)}

    def _chain(self, a, b, ex):
        """Compact path for tables: Warfarin -> CYP2C9 <- Aspirin -> hemorrhage."""
        if not ex["paths"]:
            return []
        p = ex["paths"][0]
        tail = ex["effect_tail"]
        first, last = (a, b)
        if tail and tail["drug"] == a:  # put the drug that carries the effect at the end
            first, last = b, a
        steps = [{"label": self.names[first], "type": "drug"}]
        via = p["via"]
        if p["kind"] == "ppi" and first != a:
            via = list(reversed(via))
        for k, v in enumerate(via):
            steps.append({"arrow": "→" if k == 0 else "—", "rel": p["kind"] if k == 0 else "ppi"})
            steps.append({"label": v["name"], "type": v["type"]})
        steps.append({"arrow": "←", "rel": p["kind"]})
        steps.append({"label": self.names[last], "type": "drug"})
        if tail:
            steps.append({"arrow": "→", "rel": "side_effect"})
            steps.append({"label": tail["effect"].lower(), "type": "effect"})
        return steps

    @staticmethod
    def _sentence(row, ex):
        if row["status"] == "none":
            link = ""
            if ex["paths"]:
                v = ex["paths"][0]["via"][0]["name"]
                link = (f" In the knowledge graph they are only loosely linked (through {v}), which is not "
                        f"evidence of an interaction.")
            return (f"No interaction between {row['a']['name']} and {row['b']['name']} is recorded in DrugBank or "
                    f"DDInter, and the model rates one as unlikely (probability {row['score']:.2f}).{link} "
                    f"Missing data is common, so this is not proof that the combination is safe.")
        parts = []
        if ex["paths"]:
            parts.append(ex["paths"][0]["text"])
        if row["mechanism"]:
            src = "DrugBank records that" if row["mechanism_source"] == "DrugBank" else "The model predicts that"
            m = row["mechanism"]
            parts.append(f"{src} {'t' + m[1:] if m.startswith('The ') else m}")
        if ex["effect_tail"]:
            parts.append(ex["effect_tail"]["text"])
        if not parts:
            parts.append("No shared enzyme, target, transporter, side effect or class links these two drugs in "
                         "the knowledge graph.")
        return " ".join(parts)

    def _context_alerts(self, idxs, ctx):
        flags = set()
        if ctx.get("kidney"):
            flags.add("kidney")
        if ctx.get("liver"):
            flags.add("liver")
        age = ctx.get("age")
        if age is not None and age >= 65:
            flags.add("elderly")
        alerts = []
        for i in idxs:
            r = self.drugs.iloc[i]
            text = " ".join(str(x) for x in (r["name"], r["categories"], r["atc3"], r["atc4"]) if isinstance(x, str)).lower()
            for flag, pattern, msg in CONTEXT_RULES:
                if flag in flags and re.search(pattern, text):
                    alerts.append({"drug": self.drug_card(i), "reason": flag, "message": msg, "source": "rule"})
            for cond in ctx.get("conditions") or []:
                c = cond.strip().lower()
                if not c:
                    continue
                for n, edata in self.G[i].items():
                    if any(e["rel"] == "contraindication" for e in edata.values()) and \
                            c in self.G.nodes[n]["name"].lower():
                        alerts.append({"drug": self.drug_card(i), "reason": "condition",
                                       "message": f"Listed as contraindicated in {self.G.nodes[n]['name']}.",
                                       "source": "PrimeKG contraindication"})
                        break
            for allergy in ctx.get("allergies") or []:
                al = allergy.strip().lower()
                if al and al in text:
                    alerts.append({"drug": self.drug_card(i), "reason": "allergy",
                                   "message": f"Matches the recorded allergy '{allergy}'.", "source": "rule"})
        return alerts

    def analyze(self, drug_names, context=None):
        context = context or {}
        idxs, unknown, normalised = [], [], []
        for name in drug_names:
            r = self.resolve_all(name)
            if not r["idxs"]:
                unknown.append(name)
            if r["source"] == "rxnorm":
                normalised.append({"input": name, "drugs": [self.drug_card(i) for i in r["idxs"]], "source": "RxNorm",
                                   "match": r["rxnorm"].get("match")})
            for i in r["idxs"]:
                if i not in idxs:
                    idxs.append(i)
        pairs = [self._pair(a, b) for a, b in itertools.combinations(idxs, 2)]
        alerts = self._context_alerts(idxs, context)
        flagged_drugs = {al["drug"]["idx"] for al in alerts}
        for p in pairs:
            p["context_flag"] = p["a"]["idx"] in flagged_drugs or p["b"]["idx"] in flagged_drugs
        rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        status_rank = {"known": 0, "predicted": 1, "none": 2}
        pairs.sort(key=lambda p: (rank[p["severity"]], status_rank[p["status"]], not p["context_flag"], -p["score"]))

        flagged = [p for p in pairs if p["severity"] != "none"]
        prod = 1.0
        for p in flagged:
            eff = 1.0 if p["status"] == "known" else p["score"]
            prod *= 1 - SEVERITY_WEIGHT[p["severity"]] * eff
        score = 1 - prod
        worst = min((rank[p["severity"]] for p in flagged), default=rank["none"])
        level = {0: "HIGH", 1: "MODERATE", 2: "LOW", 3: "LOW"}[worst]
        if alerts and level == "LOW":
            level = "MODERATE"
        return {
            "drugs": [self.drug_card(i) for i in idxs], "unknown": unknown, "normalised": normalised,
            "regimen": {"level": level, "score": round(score, 4), "pairs_checked": len(pairs),
                        "flagged": len(flagged)},
            "pairs": pairs, "alerts": alerts, "disclaimer": DISCLAIMER,
            "model": {"name": self.model_info["conv"].upper(), "threshold": self.threshold},
        }

    def pair_detail(self, a_id, b_id):
        a, b = self.id2idx.get(a_id), self.id2idx.get(b_id)
        if a is None or b is None or a == b:
            return None
        d = self._pair(a, b, with_detail=True)
        d["drug_info"] = {x: self._drug_info(i) for x, i in (("a", min(a, b)), ("b", max(a, b)))}
        d["model_explanation"] = self.model_explanation(a, b)
        d["disclaimer"] = DISCLAIMER
        return d

    def _drug_info(self, i):
        r = self.drugs.iloc[i]
        def s(v):
            return v if isinstance(v, str) and v else None
        return {"mechanism_of_action": s(r["mechanism_of_action"]), "categories": s(r["categories"]),
                "atc": s(r["atc4"]) or s(r["atc3"]), "n_targets": int(r["n_target"]),
                "n_enzymes": int(r["n_enzyme"]), "n_side_effects": int(r["n_side_effect"])}

    def neighbourhood(self, drug_id, highlight_with=None):
        i = self.id2idx.get(drug_id)
        if i is None:
            return None
        g = self.explainer.neighbourhood(i)
        partners = self.partners.get(i, [])
        g["counts"]["interaction partners"] = len(partners)
        g["drug"] = {**self.drug_card(i), **self._drug_info(i)}
        if highlight_with and highlight_with in self.id2idx:
            j = self.id2idx[highlight_with]
            ex = self.explainer.explain(i, j)
            ids = {n["id"] for n in g["nodes"]}
            for n in ex["graph"]["nodes"]:
                if n["id"] not in ids:
                    g["nodes"].append({**n, "focus": n["id"] == i})
            g["highlight"] = ex["graph"]["edges"]
        return g

    # ------------------------------------------------------------ reports
    @staticmethod
    def read_json(name):
        p = REPORTS / name
        return json.loads(p.read_text()) if p.exists() else None
