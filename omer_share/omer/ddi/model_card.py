"""Generate MODEL_CARD.md from reports/metrics.json and reports/eda.json.

    python -m ddi.model_card
"""
import json

from .config import DISCLAIMER, REPORTS, ROOT

NAMES = {"fp_lr": "Fingerprint LR", "fp_xgb": "Fingerprint XGBoost", "gnn_rgcn": "R-GCN", "gnn_rgat": "R-GAT",
         "gnn_rgcn_gin": "R-GCN + GIN"}
SPLITS = {"random": "random", "cold": "cold-start"}


def table(metrics, labels, part):
    rows = ["| Model | Split | AUROC | AUPRC | F1 | Hits@20 | Hits@50 | Hits@100 |",
            "|---|---|---|---|---|---|---|---|"]
    for m in NAMES:
        for s in SPLITS:
            r = metrics.get(m, {}).get(labels, {}).get(s)
            if not r:
                continue
            t = r[part]
            rows.append(f"| {NAMES[m]} | {SPLITS[s]} | {t['auroc']:.3f} | {t['auprc']:.3f} | {t['f1']:.3f} | "
                        f"{t['hits@20']:.3f} | {t['hits@50']:.3f} | {t['hits@100']:.3f} |")
    return rows


def main():
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    eda = json.loads((REPORTS / "eda.json").read_text())
    d = eda["ddi"]
    heads = ["| Model | Split | Type accuracy | Type macro-F1 | Side-effect mean AUPRC | Side-effect mean AUROC |",
             "|---|---|---|---|---|---|"]
    for m in ("gnn_rgcn", "gnn_rgat", "gnn_rgcn_gin"):
        for s in SPLITS:
            r = metrics.get(m, {}).get("merged", {}).get(s)
            if r and "type" in r:
                se = r.get("side_effects", {})
                heads.append(f"| {NAMES[m]} | {SPLITS[s]} | {r['type']['accuracy']:.3f} | {r['type']['macro_f1']:.3f} | "
                             f"{se.get('mean_auprc', float('nan')):.3f} | {se.get('mean_auroc', float('nan')):.3f} |")

    md = [
        "# Model card: drug interaction and adverse-effect predictor", "",
        "## Intended use", "",
        "Decision support for checking a medication list for drug-drug interactions, for pharmacists, clinicians "
        "and students. It is not a medical device and must not be the only basis for a prescribing decision.", "",
        f"**Disclaimer.** {DISCLAIMER}", "",
        "## Model served by the app", "",
        "- R-GCN encoder over a drug knowledge graph (PrimeKG + TDC DrugBank DDI + DDInter + TWOSIDES), trained on "
        "the merged labels with the random split. Drug nodes start from Morgan fingerprints.",
        "- Heads: interaction probability, DrugBank interaction type (86 types), TWOSIDES polypharmacy side effects "
        "(915 labels).",
        "- Severity: DDInter level when available, otherwise a rule on the interaction type (`ddi/severity.py`).",
        "- Explanation: shortest, most specific paths in the knowledge graph (`ddi/explain.py`). Explanations are "
        "grounded in real edges but are not the model's internal reasoning.",
        "- Model attributions: GNNExplainer (`ddi/model_explain.py`) learns a soft mask over the links feeding the two "
        "drugs' embeddings and reports the neighbours the score depends on (the \"What the model looked at\" card). "
        "Agreement with the path explanations is in `reports/explanation_agreement.md`.",
        "- Variant R-GCN + GIN: drug nodes start from a GIN over the molecular graph (RDKit atoms and bonds, 3 "
        "GINEConv layers, mean+max pooling) instead of fingerprints; trained with `python -m ddi.gnn --drug-encoder "
        "gin`. Not served: it matches the fingerprint R-GCN on the random split but is worse on cold-start "
        "(AUROC 0.827 vs 0.870, AUPRC 0.830 vs 0.863; Hits@K slightly higher), the split that matters for new "
        "drugs.", "",
        "## Data", "",
        f"- {d['n_drugs']:,} drugs; {d['positives']['merged']:,} known interacting pairs out of {d['possible_pairs']:,} "
        f"possible (TDC {d['positives']['tdc']:,}, PrimeKG {d['positives']['primekg']:,}, DDInter {d['positives']['ddinter']:,}).",
        f"- TWOSIDES: {eda['twosides']['pairs']:,} drug pairs with {eda['twosides']['side_effects']} side effects "
        "(at least 500 pairs each).",
        "- Unknown pairs are unlabelled, not safe. Training draws them half uniformly and half degree-matched; "
        "evaluation reports both.", "",
        "## Results: merged labels, test set, uniform unlabelled pairs", "",
        *table(metrics, "merged", "test"), "",
        "## Results: merged labels, test set, degree-matched unlabelled pairs (harder)", "",
        *table(metrics, "merged", "test_hard"), "",
        "## Results: TDC DrugBank labels only, test set, uniform unlabelled pairs", "",
        *table(metrics, "tdc", "test"), "",
        "## Extra heads (merged labels, test set)", "",
        *heads, "",
        "F1 uses the threshold that maximises F1 on validation. Hits@K is OGB-style: the share of test positives "
        "scored above the K-th highest unlabelled pair.", "",
        "## Limitations", "",
        "- **Association, not causation.** TWOSIDES and FAERS come from spontaneous reports with reporting bias.",
        "- **Missing is not safe.** Unlabelled pairs include real, unrecorded interactions, so the metrics are lower "
        "bounds and \"no interaction found\" is never evidence of safety.",
        "- **Random-split numbers are optimistic.** The cold-start split (unseen drugs) is the realistic estimate for "
        "new drugs.",
        "- **Mechanism detail.** The open data does not say whether a drug inhibits, induces or is a substrate of an "
        "enzyme, so explanations say \"both processed by CYP3A4\".",
        "- **Severity rules** for pairs that are not in DDInter are our own estimates.",
        "- **Coverage.** 1,706 DrugBank small molecules; biologics and newer drugs are mostly absent. SIDER was last "
        "updated in 2016.",
        "- **Not clinically validated.** Spot checks against DrugBank and DDInter are in the README; there has been no "
        "prospective evaluation.",
    ]
    (ROOT / "MODEL_CARD.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote MODEL_CARD.md")


if __name__ == "__main__":
    main()
