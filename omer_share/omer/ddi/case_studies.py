"""Case studies and top novel predictions for manual spot-checking.

    python -m ddi.case_studies

Writes reports/case_studies.md:
1. The spec's case studies (warfarin + NSAIDs, statin + clarithromycin, SSRIs + tramadol) and the demo regimen.
2. The highest-scoring pairs among common drugs that have *no* recorded interaction, to check by hand against
   Drugs.com or DrugBank. These are the model's novel claims.
"""
import itertools

import numpy as np
import torch

from .config import REPORTS
from .service import DDIService

CASES = [
    ("Warfarin", "Ibuprofen"), ("Warfarin", "Naproxen"), ("Warfarin", "Aspirin"),
    ("Simvastatin", "Clarithromycin"), ("Atorvastatin", "Clarithromycin"),
    ("Sertraline", "Tramadol"), ("Fluoxetine", "Tramadol"), ("Citalopram", "Tramadol"),
    ("Warfarin", "Omeprazole"), ("Simvastatin", "Omeprazole"),
]
# A pool of commonly prescribed drugs for the novel-prediction search
COMMON = ["Warfarin", "Aspirin", "Omeprazole", "Simvastatin", "Atorvastatin", "Metformin", "Lisinopril",
          "Amlodipine", "Metoprolol", "Losartan", "Levothyroxine", "Sertraline", "Fluoxetine", "Citalopram",
          "Tramadol", "Ibuprofen", "Naproxen", "Acetaminophen", "Clopidogrel", "Furosemide", "Prednisone",
          "Gabapentin", "Alprazolam", "Amoxicillin", "Azithromycin", "Ciprofloxacin", "Clarithromycin",
          "Digoxin", "Diltiazem", "Hydrochlorothiazide", "Insulin glargine", "Montelukast", "Pantoprazole",
          "Allopurinol", "Spironolactone", "Carvedilol", "Fluconazole", "Phenytoin", "Lithium", "Methotrexate"]


def main():
    s = DDIService()
    md = ["# Case studies and spot checks", "",
          f"Model: {s.model_info['file']} (threshold {s.threshold:.2f}). Severity source and evidence as shown "
          "in the app. These are qualitative checks, not a clinical validation.", "",
          "## Spec case studies", "",
          "| Pair | Status | Severity | Score | Top path | Top predicted effects |", "|---|---|---|---|---|---|"]
    for x, y in CASES:
        a, b = s.resolve(x), s.resolve(y)
        if a is None or b is None:
            md.append(f"| {x} + {y} | not in drug universe | | | | |")
            continue
        p = s._pair(a, b)
        chain = " ".join(st.get("label") or st.get("arrow") for st in p["path_chain"])
        md.append(f"| {x} + {y} | {p['status']} | {p['severity']} ({p['severity_source']}) | {p['score']:.2f} | "
                  f"{chain} | {', '.join(e['name'] for e in p['effects'][:3])} |")

    idxs = [i for i in (s.resolve(n) for n in COMMON) if i is not None]
    cand = [(a, b) for a, b in itertools.combinations(sorted(idxs), 2) if (a, b) not in s.known]
    with torch.no_grad():
        ta = torch.tensor([a for a, _ in cand])
        tb = torch.tensor([b for _, b in cand])
        probs = torch.sigmoid(s.model.ddi_logit(s.h, ta, tb)).numpy()
    order = np.argsort(-probs)[:15]
    md += ["", f"## Top novel predictions among {len(idxs)} common drugs", "",
           f"{len(cand)} of {len(idxs) * (len(idxs) - 1) // 2} pairs have no recorded interaction in DrugBank, "
           "PrimeKG or DDInter. The highest-scoring ones are listed for manual checking against Drugs.com or "
           "DrugBank; the last column is for the reviewer.", "",
           "| Pair | Score | Predicted mechanism | Top path | Reference check |", "|---|---|---|---|---|"]
    for k in order:
        a, b = cand[k]
        p = s._pair(a, b)
        chain = " ".join(st.get("label") or st.get("arrow") for st in p["path_chain"])
        md.append(f"| {s.names[a]} + {s.names[b]} | {probs[k]:.2f} | {p['mechanism'] or '–'} | {chain} | |")
    (REPORTS / "case_studies.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
