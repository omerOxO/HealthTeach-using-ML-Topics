# Drug interaction and adverse-effect predictor

A clinician or pharmacist enters a medication list. The system checks every pair, predicts whether
the drugs interact and which side effects are likely, ranks the pairs by severity, and explains each
one as a path through a biomedical knowledge graph, for example
*Warfarin → CYP2C9 ← Aspirin → hemorrhage*.

> **Decision support only.** This is a research prototype, not a medical device. A pair with no flagged
> interaction has not been shown to be safe; it may simply be untested.

**How to run and use it:** see [USER_GUIDE.md](USER_GUIDE.md), or just double-click `run.bat`
(serves the app at http://localhost:8000).

## Quick start

```bash
# 1. Python env (Python 3.12). Pick the torch build for your GPU; CPU works too.
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt

# 2. Data -> knowledge graph -> EDA -> splits
.venv\Scripts\python -m ddi.download       # ~1.7 GB: TDC, PrimeKG, DDInter
.venv\Scripts\python -m ddi.build_kg       # data/processed/* (networkx + PyG HeteroData)
.venv\Scripts\python -m ddi.eda            # reports/eda.json, reports/eda.md
.venv\Scripts\python -m ddi.splits --labels merged

# 3. Models (writes reports/metrics.json and artifacts/*)
.venv\Scripts\python -m ddi.run_all        # full grid, ~1 h on an RTX 2050
.venv\Scripts\python -m ddi.run_all --quick  # only the R-GCN the app serves

# 4. App
.venv\Scripts\python -m uvicorn api.main:app --port 8000
cd web && npm install && npm run dev       # http://localhost:5173
```

`cd web && npm run build` produces `web/dist`, which FastAPI then serves at http://localhost:8000.

## Data

| Source | Used for | How it is joined |
|---|---|---|
| TDC DrugBank DDI (DeepDDI subset) | drug universe (1,706 drugs + SMILES), 86 typed interaction labels | DrugBank IDs |
| PrimeKG (Harvard Dataverse) | drug names, targets / enzymes / transporters / carriers, SIDER side effects, indications, contraindications, off-label uses, ATC classes, protein-protein links, extra DrugBank DDIs | DrugBank IDs |
| TWOSIDES (TDC / Decagon version) | pair-level polypharmacy side effects | PubChem CID → InChIKey (connectivity block) → DrugBank |
| DDInter | Major / Moderate / Minor severity, extra DDIs | drug name → DrugBank name |

No DrugBank licence is needed: everything comes from open redistributions. PrimeKG's ATC field gives class
names, not codes, so the inhibitor / inducer / substrate distinction for enzymes is not available. Explanations
therefore say "both processed by CYP2C9" rather than "A inhibits CYP2C9".

## Labels and negatives

Missing pairs are treated as **unlabelled, not safe** (positive-unlabelled learning):

- Training samples unlabelled pairs, half uniformly and half degree-matched (endpoints drawn in proportion to
  interaction degree), so the model cannot score well by simply ranking popular drugs high.
- Evaluation uses fixed sets of unlabelled pairs, reported twice: uniform and degree-matched ("hard").
  Some of these pairs are real but unrecorded interactions, so AUPRC, F1 and Hits@K are lower bounds.
- The UI never says a pair is safe, only "no known or predicted interaction (not proof of safety)".

## Splits

- **Random:** interacting pairs shuffled 80/10/10.
- **Cold-start:** drugs shuffled 80/10/10. Test pairs contain at least one drug the model never saw interact.
  The GNN still sees that drug's targets, side effects, indications and classes, and its fingerprint.

## Models

- **Fingerprint baseline:** Morgan fingerprints (r=2, 1024 bits) of both drugs concatenated →
  logistic regression (SGD) or XGBoost. Scores are averaged over both drug orders.
- **R-GCN / R-GAT:** relational graph convolution or attention over the biological KG relations, plus a DDI
  channel (a mean over known interaction partners, computed as a dense 1.7k × 1.7k matmul). Drug nodes start
  from their fingerprints, so unseen drugs still get an embedding. The positives supervised in each batch are
  masked out of the DDI channel, so the model cannot read the answer off the graph. Three heads: interaction
  probability, DrugBank interaction type (86 classes), and TWOSIDES side effects (915-label multi-label).
- **R-GCN + GIN** (`--drug-encoder gin`): drug nodes start from a GIN (3 GINEConv layers over RDKit atoms and bonds)
  instead of fingerprints. Equal on the random split, worse on cold-start (AUROC 0.827 vs 0.870), so not served.

Metrics: AUROC, AUPRC, F1 (threshold tuned on validation), Hits@20/50/100 (OGB-style, against the unlabelled
test pairs). Side effects: mean per-label AUPRC. All results land in `reports/metrics.json` and appear on the
Analytics page.

## Severity

1. DDInter's curated level when the pair is in DDInter.
2. Otherwise a rule on the (known or predicted) DrugBank interaction type, in `ddi/severity.py`. These rules
   are our own estimates, labelled "type-based estimate" in the UI.

## Explanations

`ddi/explain.py` enumerates real KG paths between the two drugs: shared enzyme, target, transporter, carrier,
SIDER side effect, ATC class or indication, and 3-hop protein-protein bridges. Each path is scored by relation
weight × specificity, where specificity is 1 / log2(2 + the number of drugs touching the node), and the best
paths are turned into plain-language sentences. If a single-drug side effect of either drug matches a
predicted pair effect, it is added as the effect at the end of the path.

`ddi/model_explain.py` adds model-side attributions: the GNNExplainer objective (a learned soft edge mask) over the
links feeding the two drugs' R-GCN embeddings, ranked into the neighbours the score depends on. It is shown as "What
the model looked at" on the pair page and returned as `model_explanation` by `GET /api/pair`.
`python -m ddi.explain_eval` compares it with the path explanations (`reports/explanation_agreement.md`).

## Layout

```
ddi/          pipeline, models, explanations, inference service
api/main.py   FastAPI backend (/api/*)
web/          React + Vite + Cytoscape.js frontend
data/         raw downloads and processed graph (not committed)
artifacts/    trained models
reports/      metrics.json, eda.json, eda.md, run_all.log
```

## Not in this pass

A time split. Safer alternatives, FAERS evidence, prescription NER, RxNorm normalisation and the audit log with
user roles are built; see USER_GUIDE.md section 7.
