# Model card: drug interaction and adverse-effect predictor

## Intended use

Decision support for checking a medication list for drug-drug interactions, for pharmacists, clinicians and students. It is not a medical device and must not be the only basis for a prescribing decision.

**Disclaimer.** Decision support only. This tool is a research prototype, not a medical device. Predictions come from statistical models trained on public databases and can be wrong or incomplete. A pair with no flagged interaction has not been shown to be safe; it may simply be untested. Always confirm with a pharmacist, prescriber, or an authoritative reference before changing any medication.

## Model served by the app

- R-GCN encoder over a drug knowledge graph (PrimeKG + TDC DrugBank DDI + DDInter + TWOSIDES), trained on the merged labels with the random split. Drug nodes start from Morgan fingerprints.
- Heads: interaction probability, DrugBank interaction type (86 types), TWOSIDES polypharmacy side effects (915 labels).
- Severity: DDInter level when available, otherwise a rule on the interaction type (`ddi/severity.py`).
- Explanation: shortest, most specific paths in the knowledge graph (`ddi/explain.py`). Explanations are grounded in real edges but are not the model's internal reasoning.
- Model attributions: GNNExplainer (`ddi/model_explain.py`) learns a soft mask over the links feeding the two drugs' embeddings and reports the neighbours the score depends on (the "What the model looked at" card). Agreement with the path explanations is in `reports/explanation_agreement.md`.
- Variant R-GCN + GIN: drug nodes start from a GIN over the molecular graph (RDKit atoms and bonds, 3 GINEConv layers, mean+max pooling) instead of fingerprints; trained with `python -m ddi.gnn --drug-encoder gin`. Not served: it matches the fingerprint R-GCN on the random split but is worse on cold-start (AUROC 0.827 vs 0.870, AUPRC 0.830 vs 0.863; Hits@K slightly higher), the split that matters for new drugs.

## Data

- 1,706 drugs; 526,105 known interacting pairs out of 1,454,365 possible (TDC 191,402, PrimeKG 457,564, DDInter 94,569).
- TWOSIDES: 50,150 drug pairs with 915 side effects (at least 500 pairs each).
- Unknown pairs are unlabelled, not safe. Training draws them half uniformly and half degree-matched; evaluation reports both.

## Results: merged labels, test set, uniform unlabelled pairs

| Model | Split | AUROC | AUPRC | F1 | Hits@20 | Hits@50 | Hits@100 |
|---|---|---|---|---|---|---|---|
| Fingerprint LR | random | 0.700 | 0.681 | 0.693 | 0.003 | 0.006 | 0.011 |
| Fingerprint LR | cold-start | 0.613 | 0.597 | 0.670 | 0.001 | 0.001 | 0.003 |
| Fingerprint XGBoost | random | 0.884 | 0.882 | 0.808 | 0.047 | 0.080 | 0.124 |
| Fingerprint XGBoost | cold-start | 0.780 | 0.774 | 0.729 | 0.010 | 0.015 | 0.025 |
| R-GCN | random | 0.992 | 0.991 | 0.960 | 0.134 | 0.326 | 0.538 |
| R-GCN | cold-start | 0.870 | 0.863 | 0.802 | 0.025 | 0.035 | 0.049 |
| R-GAT | random | 0.992 | 0.991 | 0.958 | 0.191 | 0.345 | 0.499 |
| R-GAT | cold-start | 0.842 | 0.844 | 0.772 | 0.020 | 0.040 | 0.066 |
| R-GCN + GIN | random | 0.992 | 0.992 | 0.959 | 0.147 | 0.346 | 0.573 |
| R-GCN + GIN | cold-start | 0.827 | 0.830 | 0.758 | 0.033 | 0.045 | 0.062 |

## Results: merged labels, test set, degree-matched unlabelled pairs (harder)

| Model | Split | AUROC | AUPRC | F1 | Hits@20 | Hits@50 | Hits@100 |
|---|---|---|---|---|---|---|---|
| Fingerprint LR | random | 0.595 | 0.584 | 0.652 | 0.002 | 0.003 | 0.006 |
| Fingerprint LR | cold-start | 0.575 | 0.566 | 0.665 | 0.000 | 0.001 | 0.002 |
| Fingerprint XGBoost | random | 0.783 | 0.793 | 0.726 | 0.033 | 0.050 | 0.069 |
| Fingerprint XGBoost | cold-start | 0.736 | 0.736 | 0.701 | 0.006 | 0.012 | 0.019 |
| R-GCN | random | 0.989 | 0.989 | 0.950 | 0.213 | 0.370 | 0.520 |
| R-GCN | cold-start | 0.843 | 0.837 | 0.780 | 0.023 | 0.032 | 0.043 |
| R-GAT | random | 0.988 | 0.988 | 0.947 | 0.231 | 0.394 | 0.545 |
| R-GAT | cold-start | 0.811 | 0.818 | 0.746 | 0.017 | 0.030 | 0.051 |
| R-GCN + GIN | random | 0.989 | 0.989 | 0.948 | 0.230 | 0.373 | 0.554 |
| R-GCN + GIN | cold-start | 0.790 | 0.798 | 0.729 | 0.030 | 0.040 | 0.056 |

## Results: TDC DrugBank labels only, test set, uniform unlabelled pairs

| Model | Split | AUROC | AUPRC | F1 | Hits@20 | Hits@50 | Hits@100 |
|---|---|---|---|---|---|---|---|
| Fingerprint LR | random | 0.765 | 0.737 | 0.734 | 0.011 | 0.020 | 0.038 |
| Fingerprint LR | cold-start | 0.623 | 0.587 | 0.661 | 0.000 | 0.001 | 0.002 |
| Fingerprint XGBoost | random | 0.937 | 0.933 | 0.865 | 0.104 | 0.192 | 0.268 |
| Fingerprint XGBoost | cold-start | 0.850 | 0.850 | 0.774 | 0.033 | 0.050 | 0.084 |
| R-GCN | random | 0.997 | 0.997 | 0.982 | 0.557 | 0.848 | 0.931 |
| R-GCN | cold-start | 0.914 | 0.917 | 0.840 | 0.069 | 0.107 | 0.156 |
| R-GAT | random | 0.998 | 0.997 | 0.984 | 0.715 | 0.885 | 0.954 |
| R-GAT | cold-start | 0.919 | 0.925 | 0.847 | 0.110 | 0.166 | 0.230 |

## Extra heads (merged labels, test set)

| Model | Split | Type accuracy | Type macro-F1 | Side-effect mean AUPRC | Side-effect mean AUROC |
|---|---|---|---|---|---|
| R-GCN | random | 0.969 | 0.930 | 0.266 | 0.815 |
| R-GCN | cold-start | 0.774 | 0.730 | 0.174 | 0.722 |
| R-GAT | random | 0.966 | 0.925 | 0.259 | 0.809 |
| R-GAT | cold-start | 0.671 | 0.218 | 0.132 | 0.658 |
| R-GCN + GIN | random | 0.971 | 0.944 | 0.267 | 0.816 |
| R-GCN + GIN | cold-start | 0.656 | 0.279 | 0.133 | 0.644 |

F1 uses the threshold that maximises F1 on validation. Hits@K is OGB-style: the share of test positives scored above the K-th highest unlabelled pair.

## Limitations

- **Association, not causation.** TWOSIDES and FAERS come from spontaneous reports with reporting bias.
- **Missing is not safe.** Unlabelled pairs include real, unrecorded interactions, so the metrics are lower bounds and "no interaction found" is never evidence of safety.
- **Random-split numbers are optimistic.** The cold-start split (unseen drugs) is the realistic estimate for new drugs.
- **Mechanism detail.** The open data does not say whether a drug inhibits, induces or is a substrate of an enzyme, so explanations say "both processed by CYP3A4".
- **Severity rules** for pairs that are not in DDInter are our own estimates.
- **Coverage.** 1,706 DrugBank small molecules; biologics and newer drugs are mostly absent. SIDER was last updated in 2016.
- **Not clinically validated.** Spot checks against DrugBank and DDInter are in the README; there has been no prospective evaluation.