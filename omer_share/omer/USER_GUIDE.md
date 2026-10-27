# Interaction Check: user guide

A drug-interaction and adverse-effect predictor. You enter a patient's medication list; the system checks
every pair of drugs, says whether they are known or predicted to interact, how severe that is, which side
effects are likely, and **why**, as a path through a biomedical knowledge graph
(for example *Aspirin → gastrointestinal hemorrhage ← Warfarin → subdural hemorrhage*).

> **Decision support only.** This is a research prototype, not a medical device. Predictions can be wrong or
> incomplete. A pair with nothing flagged has **not** been shown to be safe; it may simply be untested.
> Always confirm with a pharmacist, prescriber or an authoritative reference.

---

## 1. Running the application

### Quick start (everything already set up on this machine)

Double-click **`run.bat`** in `D:\test\omer`, or from a terminal:

```bash
run.bat
```

It builds the web app if needed, starts the server, and opens **http://localhost:8000** in your browser.
The first page load takes about 10 seconds while the model and knowledge graph load. Press `Ctrl+C` in the
terminal to stop.

### Development mode (live reload while editing the UI)

Two terminals:

```bash
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```

```bash
cd web && npm run dev
```

Then open **http://localhost:5173**. The web dev server forwards `/api` calls to port 8000.

### First-time setup on a new machine

Needs Python 3.12, Node 18+, about 3 GB of disk, and an internet connection for the downloads. An NVIDIA GPU
makes training faster but is optional.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
cd web && npm install && cd ..
```

(The second line installs the scispaCy NER model. It is optional, like the transformer NER model, which downloads
itself from Hugging Face on first use.)

(For CPU only, install `torch==2.5.1` without the `--index-url` part.)

### Rebuilding the data and models

Run these in order. Each step writes files the next one reads.

| Step | Command | Output | Time (RTX 2050) |
|---|---|---|---|
| 1. Download sources | `.venv\Scripts\python -m ddi.download` | `data/raw/` (~1.7 GB) | 10–20 min, network-bound |
| 2. Build the knowledge graph | `.venv\Scripts\python -m ddi.build_kg` | `data/processed/` | ~3 min |
| 3. Exploratory analysis | `.venv\Scripts\python -m ddi.eda` | `reports/eda.json`, `reports/eda.md` | seconds |
| 4. Train/test splits | `.venv\Scripts\python -m ddi.splits --labels merged` | `data/processed/split_*.npz` | seconds |
| 5a. Train the model the app uses | `.venv\Scripts\python -m ddi.run_all --quick` | `artifacts/`, `reports/metrics.json` | ~5 min |
| 5b. Or train every model | `.venv\Scripts\python -m ddi.run_all` | same, all models | ~30–40 min |
| 6. Model card | `.venv\Scripts\python -m ddi.model_card` | `MODEL_CARD.md` | seconds |
| 7. Case studies | `.venv\Scripts\python -m ddi.case_studies` | `reports/case_studies.md` | ~10 s |
| 8. Explanation agreement | `.venv\Scripts\python -m ddi.explain_eval` | `reports/explanation_agreement.md` | ~3 min |

Single models can also be trained on their own:

```bash
.venv\Scripts\python -m ddi.baseline --labels merged --split cold --model xgb
```

```bash
.venv\Scripts\python -m ddi.gnn --labels merged --split random --conv rgat
.venv\Scripts\python -m ddi.gnn --labels merged --split all --conv rgcn --drug-encoder gin
```

`--drug-encoder gin` starts drug nodes from a GIN over each drug's molecular graph instead of its fingerprint
(model name `gnn_rgcn_gin`, file `artifacts/gnn_rgcn_gin_<labels>_<split>.pt`). The molecule graphs are built
from the SMILES once and cached in `data/processed/mol_graphs.pt`.

Options: `--labels merged|tdc`, `--split random|cold|all`, `--model lr|xgb|all`, `--conv rgcn|rgat|all`.

---

## 2. How to operate the system

### Step 1: enter the medication list (Check page)

1. Type a medication into the search box. Suggestions appear as you type; generic names and common brand names
   both work (Coumadin → Warfarin, Zocor → Simvastatin, Aspirin → Acetylsalicylic acid, Vitamin C → Ascorbic acid).
2. Press **Enter** or click a suggestion. The drug appears as a chip below the box. Click **×** on a chip to remove it.
3. Add at least two drugs (up to 20). The button shows how many pairs will be checked.
4. To try the system quickly, click **Try the example** to load Warfarin, Aspirin, Omeprazole and Simvastatin.

**Paste prescription text tab:** paste a prescription or discharge note and click **Find medications**. A biomedical
NER model finds the medications (brand names included) and reads each one's dose, frequency, route and duration;
the drugs are added to the list and shown in a table under the text. Check the list afterwards (see section 7).

Brand, salt and misspelt names that are not in the local list are looked up in **RxNorm** (for example Lopid →
Gemfibrozil, warfarin sodium → Warfarin). The search box marks these "(RxNorm)".

### Step 2: add patient context (optional)

The **Patient context** panel re-ranks and flags warnings. Nothing is stored on the server.

| Field | What it does |
|---|---|
| Age | 65 or over flags blood thinners (bleeding risk) and sedatives/anticholinergics (Beers-list style) |
| Reduced kidney function | Flags NSAIDs, metformin, lithium, aminoglycosides |
| Reduced liver function | Flags statins, acetaminophen, drugs listed as hepatotoxic |
| Existing conditions | Matched against contraindications in the knowledge graph (e.g. "peptic ulcer") |
| Allergies | Matched against drug names and classes (e.g. "penicillin" flags amoxicillin) |

Click **Analyse**.

### Step 3: read the results (Results page)

- **Regimen risk:** HIGH, MODERATE or LOW, from the worst severity among flagged pairs.
- **Combined risk score:** 0 to 1, combining every flagged pair weighted by severity and probability.
- **Pairs checked / Flagged:** how many pairs were scored and how many are known or predicted interactions.
- **Patient-context alerts:** appear if any context rule matched.
- **Why … is the top concern:** the explanation path and a plain-language paragraph for the worst pair.
- **Drug pairs table:** one row per pair, ranked by severity, then evidence, then risk score. Click a column
  header to re-sort. Click a row for full details.

How to read a row:

| Column | Meaning |
|---|---|
| Severity | Major (red ▲), Moderate (orange ◆), Minor (amber ●), None found (green ✓). The line underneath says where it came from: **DDInter** (curated) or **type-based estimate** (our rule on the DrugBank interaction type). |
| Evidence | **Known** = listed in DrugBank, PrimeKG or DDInter. **Predicted** = not listed, but the model's probability is above the decision threshold. **None found** = neither; not proof of safety. |
| Risk | The model's probability that the pair interacts (0–1). |
| Top predicted effects | Side effects the model expects for this combination, ranked by how far they exceed their usual rate. Effects implied by the mechanism (e.g. bleeding for anticoagulants) come first. |
| Mechanism | The DrugBank interaction description, or the model's predicted one (marked *predicted*). |

### Step 4: inspect a pair (Pair details page)

- **What is going on:** the path and the plain-language explanation.
- **Explanation path:** an interactive mini-graph. The two drugs are on the left and right, the things they share
  are in the middle, and the top path is highlighted in red. Drag nodes and scroll to zoom.
- **Ranked mechanisms:** every connecting path, best first: shared metabolising enzymes, targets,
  transporters, carrier proteins, side effects, drug classes and protein-protein bridges. A protein shared by a
  few drugs ranks above one shared by hundreds.
- **What the model looked at:** the neighbours (proteins, side effects, diseases, classes, interaction partners)
  whose links the model needed to keep its score, from GNNExplainer. Each row says whether the neighbour belongs to
  one drug or both. **also a KG path** marks neighbours the path explanation uses too. The small table shows the
  score with only those neighbours' links and with all links except theirs: if the first stays close to the full
  score, the attribution is faithful. The first visit to a pair takes about a second while this is computed.
- **Predicted side effects:** probability bars for the top 15 effects. **TWOSIDES** marks effects already reported
  for this exact pair.
- **Evidence:** what DrugBank, PrimeKG, DDInter and TWOSIDES say about the pair, the FAERS report counts, and
  safer alternatives for a flagged pair (section 7).
- **Drug cards:** class, counts of targets, enzymes and side effects, and mechanism of action for each drug.

### Step 5: explore the knowledge graph (Graph explorer)

- Search for any drug to see its neighbourhood: targets and enzymes (grey), side effects (red), diseases (teal) and
  drug classes (amber). Up to 12 of the most specific neighbours per type are shown.
- Use **Compare with a second drug** to highlight the paths connecting the two drugs.
- Click a node to see its name and type. The table on the right counts all neighbours, including the number of
  known interaction partners.

### Step 6: check model quality (Analytics page)

- **Filters:** labels (merged or TDC-only), metric (AUROC, AUPRC, F1, Hits@20/50/100), and unlabelled-pair type
  (uniform or degree-matched, which is harder).
- **Model comparison chart:** red = random split, hatched = cold-start split (drugs never seen in training).
  Hover a group for the values, or click **Show table** for every number.
- **GNN extra heads:** interaction-type accuracy and side-effect AUPRC.
- **EDA:** graph size, degree distribution, most connected drugs, most frequent side effects, most common
  interaction types, cold-start exposure and where the labels come from.

### Step 7: model card

Intended use, data, full result tables, and limitations. Read this before relying on any output.

### Theme

The sun/moon button in the top-right corner switches between dark (default) and light. The choice is remembered.

---

## 3. What each feature is and how it works

### Data pipeline and knowledge graph (`ddi/build_kg.py`)

| Source | What it adds | How it is joined |
|---|---|---|
| TDC DrugBank DDI | the 1,706 drugs, their structures (SMILES) and 191k interactions labelled with 86 types | DrugBank ID |
| PrimeKG | drug names; targets, enzymes, transporters and carriers; SIDER side effects; indications, contraindications and off-label uses; ATC classes; protein-protein links; 458k more DrugBank interactions | DrugBank ID |
| DDInter | Major/Moderate/Minor severity for 95k pairs | drug name |
| TWOSIDES | 915 polypharmacy side effects for 50k drug pairs | chemical structure (InChIKey) |

The result is a graph of 7,456 nodes (drugs, proteins, side effects, diseases, drug classes) and 117k edges,
saved as a networkx graph and a PyTorch Geometric graph. 526k drug pairs are known to interact, about 36% of
all possible pairs. Drug-drug interactions are kept separate as *labels*, so the model never sees the answers
as graph edges.

### Labels and "missing is not safe"

Pairs without a recorded interaction are treated as **unlabelled**, not safe. Training compares known
interactions with a mix of random and degree-matched unlabelled pairs, so the model cannot just score popular
drugs high. The app never says "safe", only "no interaction found (not proof of safety)".

### Models

| Model | Idea |
|---|---|
| Fingerprint baseline (LR, XGBoost) | Each drug's molecular fingerprint (Morgan, 1,024 bits); the two are concatenated and classified. |
| R-GCN (served by the app) | A graph neural network that passes messages along each relation type (enzyme, target, side effect …) plus known interactions. Each drug's representation then reflects its biology. Drugs start from their fingerprints, so new drugs still work. |
| R-GAT | The same, with attention: each relation type weighs its neighbours. |
| R-GCN + GIN | The R-GCN, but each drug starts from a small graph network (GIN) run over its molecule (atoms and bonds) instead of its fingerprint. Not served: on unseen drugs it scores lower than the fingerprint version. |

The graph models have three outputs: interaction probability, which of the 86 DrugBank interaction types, and
probabilities for 915 polypharmacy side effects.

### Splits and results

- **Random split:** pairs are shuffled 80/10/10. Numbers are optimistic.
- **Cold-start split:** test pairs involve drugs whose interactions the model never saw. This is the realistic
  estimate for a new drug.

Test-set results, merged labels, uniform unlabelled pairs:

| Model | Split | AUROC | AUPRC | F1 | Hits@20 | Hits@100 |
|---|---|---|---|---|---|---|
| Fingerprint LR | random | 0.700 | 0.681 | 0.693 | 0.003 | 0.011 |
| Fingerprint LR | cold-start | 0.613 | 0.597 | 0.670 | 0.001 | 0.003 |
| Fingerprint XGBoost | random | 0.884 | 0.882 | 0.808 | 0.047 | 0.124 |
| Fingerprint XGBoost | cold-start | 0.780 | 0.774 | 0.729 | 0.010 | 0.025 |
| R-GCN | random | 0.992 | 0.991 | 0.960 | 0.134 | 0.538 |
| R-GCN | cold-start | 0.870 | 0.863 | 0.802 | 0.025 | 0.049 |
| R-GAT | random | 0.992 | 0.991 | 0.958 | 0.191 | 0.499 |
| R-GAT | cold-start | 0.842 | 0.844 | 0.772 | 0.020 | 0.066 |
| R-GCN + GIN | random | 0.992 | 0.992 | 0.959 | 0.147 | 0.573 |
| R-GCN + GIN | cold-start | 0.827 | 0.830 | 0.758 | 0.033 | 0.062 |

TDC-only labels, degree-matched negatives and the extra heads are in [MODEL_CARD.md](MODEL_CARD.md) and on the
Analytics page.

### Severity (`ddi/severity.py`)

1. If DDInter lists the pair, its Major/Moderate/Minor level is used.
2. Otherwise a rule on the interaction type is used. For example, "increases anticoagulant activity", QT
   prolongation and serotonergic effects are Major; "photosensitising" is Minor. These rules are our own
   estimates and are labelled as such.

### Explanations (`ddi/explain.py`)

For each pair the system lists every real path between the two drugs in the knowledge graph and ranks them:

- **Relation type:** shared enzymes, transporters and targets count most.
- **Specificity:** a node shared by few drugs counts more.
- **Mechanism fit:** for pharmacokinetic interactions ("serum concentration increased") shared enzymes and
  transporters are boosted; for an anticoagulant interaction, bleeding-type side effects are boosted.

The best path becomes the sentence and the highlighted graph. Because explanations are built from real database
edges, you can check them; they are not the model's internal reasoning.

### Model attributions (`ddi/model_explain.py`)

The **What the model looked at** card shows what the R-GCN itself relied on. It uses the GNNExplainer objective:
a soft on/off weight is learned for every link feeding the two drugs' embeddings (their knowledge-graph links and
their known-interaction links, in both layers), so that the masked graph keeps the model's prediction while as
few links as possible stay on. Link weights are summed per neighbour and ranked. PyG's own `Explainer` cannot mask
`RGCNConv` (it propagates one relation at a time) or the dense interaction channel, so the two layers are
re-implemented for the two drugs; the code checks that with every link on it reproduces the model's score exactly.
How often these attributions agree with the path explanations is measured by `ddi/explain_eval.py` and written to
`reports/explanation_agreement.md`.

### Patient context (`ddi/service.py`)

Simple, transparent rules (listed in Step 2) plus contraindication lookups in the knowledge graph. Flagged drugs
move up within their severity group and are tagged *patient context*.

### API (FastAPI, `api/main.py`)

Interactive documentation lives at **http://localhost:8000/docs**.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status and model information |
| `GET /api/drugs/search?q=war` | autocomplete |
| `POST /api/analyze` | `{"drugs": ["Warfarin","Aspirin"], "context": {"age": 70, "kidney": true}}` → pairs, severity, explanations, regimen risk |
| `POST /api/extract` | `{"text": "..."}` → medications found in free text, each with dose, frequency, route, duration and source |
| `GET /api/alternatives?a=DB00682&b=DB00945&meds=…` | safer same-class / same-indication alternatives for a pair, given the rest of the list |
| `GET /api/faers/pair?a=DB00682&b=DB00945` | FAERS co-report counts, top reactions and yearly counts from openFDA |
| `GET /api/faers/overview?ids=DB00682,DB00945,…` | FAERS co-report counts for every pair of up to 10 drugs |
| `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` | local sign-in (session cookie) |
| `GET /api/audit` | audit log (administrators only) |
| `GET /api/pair?a=DB00682&b=DB00945` | full pair details, including `model_explanation` (GNNExplainer top neighbours and edges, fidelity scores) |
| `GET /api/graph/drug/DB00682?highlight=DB00945` | graph neighbourhood |
| `GET /api/metrics`, `/api/eda`, `/api/model-card` | data for the Analytics and Model card pages |

---

## 4. Project layout

```
run.bat           one-click start
ddi/              pipeline, models, explanations, inference service
  download.py build_kg.py eda.py splits.py baseline.py gnn.py run_all.py
  explain.py severity.py service.py metrics.py model_card.py case_studies.py
  alternatives.py faers.py ner.py rxnorm.py audit.py   (section 7)
api/main.py       FastAPI backend
web/              React + Vite + Cytoscape.js frontend
data/             raw downloads and processed graph (not in git); data/cache/ (RxNorm, FAERS), data/audit.sqlite
artifacts/        trained models (not in git)
reports/          metrics.json, eda.md, case_studies.md, run_all.log
MODEL_CARD.md     results and limitations
```

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| Page shows "500" or empty results right after starting | The model is still loading; wait about 10 s and refresh. |
| `run.bat` says the trained model was not found | Run `.venv\Scripts\python -m ddi.run_all --quick`. |
| A drug is "not recognised" | Try the generic name. The system covers 1,706 DrugBank small-molecule drugs; most biologics and very new drugs are not included. |
| Port 8000 or 5173 already in use | Close the other server, or pass another port: `--port 8001`. |
| Training runs out of GPU memory | Train on CPU (slower), or lower `--hidden` (e.g. `--hidden 64`). |

## 6. Known limitations

- TWOSIDES and FAERS-type data show statistical association, not causation, and partly reflect the condition
  being treated, so some predicted effects (e.g. heart attack) are confounded.
- The open data does not say whether a drug inhibits, induces or is broken down by an enzyme, so
  explanations say "both processed by CYP2C9".
- Severity for pairs not in DDInter is our own estimate.
- Not clinically validated. `reports/case_studies.md` lists spot checks and the model's top new predictions to
  check by hand against Drugs.com or DrugBank.

## 7. Safer alternatives, FAERS, prescription NER, RxNorm and the audit log

These five features build on the core pipeline. FAERS and RxNorm call free public APIs (no key needed), so they
need an internet connection the first time a name or pair is looked up; after that the answers come from a local
cache in `data/cache/`.

### Safer alternatives (`ddi/alternatives.py`, Pair details → Evidence → *Safer alternatives*)

For a flagged pair, the system looks for drugs that could do the same job as either drug and interact less with the
rest of the analysed list.

1. **Candidates:** approved, not-withdrawn drugs with the same main drug class (the class shown in the app) or a
   large overlap in knowledge-graph indications (at least two shared, Jaccard ≥ 0.25).
2. **Scoring:** each candidate takes the drug's place in the list and is checked against every other drug with the
   same logic as the main analysis (known or predicted interaction, severity, probability).
3. **Kept:** only candidates whose worst severity is no worse and whose combined risk is lower. Up to five per drug,
   least severe first.

Example: for Warfarin + Aspirin in the four-drug example, nothing in Warfarin's class does better (every
anticoagulant interacts with aspirin), while for aspirin it lists drugs such as salsalate and dipyridamole, each with
a moderate rather than major worst interaction. Hover an entry for the full reason.

**Limits:** dose, efficacy and the patient's actual indication are not checked. This is a list to discuss with the
prescriber, not a substitution. The suggestions use the whole list analysed on the Check page; opening a pair page
without an analysis checks the two drugs against each other only.

### FAERS evidence (`ddi/faers.py`, Pair details → Evidence → *FAERS report count*, and the **FAERS reports** page)

Counts come from the FDA Adverse Event Reporting System through the openFDA API (`api.fda.gov/drug/event.json`).

- **Names:** DrugBank names are mapped to the RxNorm ingredient name that openFDA uses (Acetylsalicylic acid →
  aspirin), then searched in `patient.drug.openfda.generic_name`. Salts and combination products that contain the
  drug are included. The page shows exactly which names and query were used.
- **Numbers:** reports listing both drugs, how many were serious or fatal, a **co-report ratio** (observed ÷ the
  number expected if the two drugs were reported independently), the most reported reactions, and reports per year.
- **"× alone"** marks a reaction that makes up a larger share of the pair's reports than of either drug's own
  reports (1.5× or more). For Warfarin + Aspirin, gastrointestinal haemorrhage is about 1.8×.
- **FAERS reports page:** pick any two drugs, or use the table of co-report counts for every pair in your current
  list (first 10 drugs). Click **Show table** on a chart for the numbers.
- **Cache:** `data/cache/faers.json`, refreshed after 30 days. Without an API key openFDA allows about 1,000 requests
  a day; a pair costs 10 the first time. To raise the limit, set `OPENFDA_API_KEY` before starting the server.

**Reporting bias:** FAERS is voluntary, spontaneous reporting. A report naming both drugs does not mean the
combination caused the reaction. Counts follow how widely a drug is used, publicity, litigation and who reports;
they are not incidence rates, and most events are never reported. Use FAERS as context, not as proof.

### Prescription text: NER with dose and frequency (`ddi/ner.py`, Check → *Paste prescription text*)

1. Two biomedical NER models read the text: **d4data/biomedical-ner-all** (a transformer trained on clinical case
   reports: medication, dosage, frequency, route and duration spans) and **scispaCy en_ner_bc5cdr_md** (chemical
   names). The local name list (DrugBank names and the brand-name map) adds anything the models miss.
2. Each medication is matched to a DrugBank drug: first the local list, then RxNorm. "Metoprolol succinate ER"
   becomes Metoprolol.
3. Dose, frequency, route and duration are read from the text after each medication, up to the next medication or
   the end of the line. Abbreviations are expanded: qhs → at bedtime, BID → twice daily, q6h → every 6 hours,
   PO → oral, prn → as needed.

The result shows the text with medications highlighted, a table (in the text, medication, dose, frequency, route,
duration and which model found it), and any medication names that matched nothing in the database.

The first extraction after start-up can take up to 30 seconds while the models load (the server starts loading them
in the background). Doses and frequencies are shown for checking only; the interaction analysis does not use them.
Set `DDI_NER=off` to use the name list only.

### RxNorm normalisation (`ddi/rxnorm.py`)

When a typed or pasted name is not in the local list, the system asks the RxNav REST API (`rxnav.nlm.nih.gov`):
exact or normalised match first, then RxNav's approximate match for misspellings. It takes the RxNorm ingredient(s)
and finds them locally, stripping salt words if needed (warfarin sodium → Warfarin). A combination brand adds every
ingredient it can find (Advair → Salmeterol and Fluticasone propionate).

- The `SYNONYMS` brand-name map in `ddi/service.py` is still checked first, so common names work offline.
- Every answer, including "not found", is cached in `data/cache/rxnorm.json`. Network failures are retried after a day.
- The Results page lists names normalised this way. **"Closest match, please check"** means RxNorm's approximate
  match was used (for example *lipitorr* → Atorvastatin).
- The autocomplete only uses exact RxNorm names, to avoid fuzzy matches while you are still typing.
- Set `DDI_OFFLINE=1` to turn RxNorm lookups off.

### Audit log and user roles (`ddi/audit.py`, **Audit log** page)

Every analysis is written to `data/audit.sqlite`: time (UTC), the drug list, number of unrecognised names, pairs
checked and flagged, regimen risk level and score, the top five flagged pairs, the model, the role of whoever ran it,
and which *kinds* of patient context were filled in (for example "age, kidney"). Patient values, conditions,
allergies, usernames and IP addresses are **not** stored.

| Role | Can |
|---|---|
| guest (not signed in) | run analyses (logged as "guest") |
| clinician | the same, logged as "clinician" |
| admin | also open the **Audit log** page and filter it by risk level |

Sign in with **Sign in** at the top right (or on the Audit log page). Two local accounts are created on first start:

| Username | Default password | Change it with |
|---|---|---|
| `admin` | `admin` | environment variable `DDI_ADMIN_PASSWORD` |
| `clinician` | `clinician` | environment variable `DDI_CLINICIAN_PASSWORD` |

The variables only take effect when the account is first created. To reset the accounts, stop the server, delete
`data/audit.sqlite` (this also deletes the log), set the variables and start again. Passwords are stored as salted
PBKDF2 hashes; sessions last 8 hours and end when the server restarts. **This is a prototype login:** change the
default passwords before anyone else can reach the app, and do not expose it to a network.

### Configuration summary

| Variable | Effect |
|---|---|
| `OPENFDA_API_KEY` | optional openFDA key (higher daily limit) |
| `DDI_OFFLINE=1` | no RxNorm lookups (FAERS still needs the network) |
| `DDI_NER=off` | prescription text uses the name list only |
| `DDI_ADMIN_PASSWORD`, `DDI_CLINICIAN_PASSWORD` | passwords for the seed accounts (first start only) |
