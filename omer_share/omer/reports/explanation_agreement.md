# Model attributions vs knowledge-graph path explanations

Model: gnn_rgcn_merged_random.pt. Model attributions come from GNNExplainer (`ddi/model_explain.py`): a learned soft mask over the edges feeding the two drugs' embeddings (knowledge-graph links and known-interaction links), ranked into neighbour nodes. Path explanations come from `ddi/explain.py`: shared enzymes, targets, transporters, classes, side effects and protein-protein bridges, ranked by relation weight and specificity. Both rank neighbour nodes of the two drugs, so we compare the rankings.

## Summary

| Measure | Case studies (10 pairs) | Random known test pairs (200) |
|---|---|---|
| pairs | 10 | 200 |
| with a KG path | 10 | 155 |
| overlap@1 | 0.00 | 0.19 |
| random@1 | 0.001 | 0.001 |
| overlap@3 | 0.10 | 0.15 |
| random@3 | 0.002 | 0.002 |
| overlap@5 | 0.16 | 0.11 |
| random@5 | 0.003 | 0.003 |
| path top in model top-1 | 0.00 | 0.19 |
| path top in model top-3 | 0.00 | 0.23 |
| path top in model top-5 | 0.00 | 0.25 |
| model top on a path | 0.80 | 0.37 |
| score, all links | 0.79 | 0.80 |
| score, only top-8 links | 0.86 | 0.79 |
| score, all but top-8 | 0.77 | 0.61 |
| seconds per explanation | 0.81 | 0.66 |

`overlap@k` is the share of the model's top-k neighbours that are also among the top-k path nodes; `random@k` is the overlap that k neighbours picked at random from all neighbours of the two drugs would get. The score rows are mean predicted probabilities: with only the top-8 neighbours' links kept, and with every link except theirs.

## Case studies

| Pair | Status | Score | Model's top neighbour | Best KG path node | Overlap@3 | Overlap@5 | Model top on a path | Score, only top-8 |
|---|---|---|---|---|---|---|---|---|
| Warfarin + Ibuprofen | known | 0.76 | CYP2C9 | Gastrointestinal hemorrhage | 0.00 | 0.20 | yes | 0.84 |
| Warfarin + Naproxen | known | 0.76 | Antithrombotic agents | Subdural hemorrhage | 0.00 | 0.00 | no | 0.86 |
| Warfarin + Acetylsalicylic acid | known | 0.76 | Antithrombotic agents | Gastrointestinal hemorrhage | 0.00 | 0.20 | yes | 0.89 |
| Simvastatin + Clarithromycin | known | 0.83 | CYP3A5 | SLCO1B3 | 0.33 | 0.20 | yes | 0.90 |
| Atorvastatin + Clarithromycin | known | 0.85 | CYP3A5 | SLCO1B3 | 0.33 | 0.20 | yes | 0.93 |
| Sertraline + Tramadol | known | 0.76 | CYP2D6 | Seizure | 0.00 | 0.20 | yes | 0.85 |
| Fluoxetine + Tramadol | known | 0.75 | CYP2D6 | Seizure | 0.00 | 0.20 | yes | 0.82 |
| Citalopram + Tramadol | known | 0.76 | CYP2D6 | Seizure | 0.00 | 0.20 | yes | 0.85 |
| Warfarin + Omeprazole | known | 0.82 | CYP1A2 | CYP2C18 | 0.33 | 0.20 | yes | 0.82 |
| Simvastatin + Omeprazole | known | 0.83 | Combinations for eradication of helicobacter pylori | CYP2C8 | 0.00 | 0.00 | no | 0.83 |

## Reading the result

- On the case studies, the model's top-5 neighbours overlap the top-5 path nodes 16% of the time, 50x what random neighbours would give; the model's top neighbour lies on a path explanation in 80% of pairs.
- On the random test pairs, the model's top-5 neighbours overlap the top-5 path nodes 11% of the time, 35x what random neighbours would give; the model's top neighbour lies on a path explanation in 37% of pairs.
- Keeping only the top-8 neighbours' links keeps the mean score at 0.79 (vs 0.80 with every link), so the attribution captures what the model uses. Removing them lowers it only to 0.61: the remaining links carry much of the same signal (many neighbours are redundant), so the attributions are sufficient rather than necessary.
- Disagreements are expected: the path explainer only looks at neighbours the two drugs share (or bridge through a protein link), while the model can also lean on one drug's own neighbours (its class, indications or a single target) that tell it the drug is interaction-prone. Those show up in the app as neighbours of one drug only, without the "also a KG path" tag.