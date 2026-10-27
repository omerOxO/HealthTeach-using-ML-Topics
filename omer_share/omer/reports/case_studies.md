# Case studies and spot checks

Model: gnn_rgcn_merged_random.pt (threshold 0.45). Severity source and evidence as shown in the app. These are qualitative checks, not a clinical validation.

## Spec case studies

| Pair | Status | Severity | Score | Top path | Top predicted effects |
|---|---|---|---|---|---|
| Warfarin + Ibuprofen | known | major (DDInter) | 0.76 | Warfarin → Gastrointestinal hemorrhage ← Ibuprofen → intraventricular hemorrhage | haematoma, Vitreous haemorrhage, haemorrhage rectum |
| Warfarin + Naproxen | known | major (DDInter) | 0.76 | Warfarin → Subdural hemorrhage ← Naproxen → gastrointestinal hemorrhage | haematoma, Vitreous haemorrhage, Bleeding |
| Warfarin + Aspirin | known | major (DDInter) | 0.76 | Acetylsalicylic acid → Gastrointestinal hemorrhage ← Warfarin → subdural hemorrhage | Vitreous haemorrhage, haematoma, haemorrhage rectum |
| Simvastatin + Clarithromycin | known | major (DDInter) | 0.83 | Simvastatin → SLCO1B3 ← Clarithromycin | Mod, arteriosclerotic heart disease, elevated cholesterol |
| Atorvastatin + Clarithromycin | known | major (DDInter) | 0.85 | Atorvastatin → SLCO1B3 ← Clarithromycin | arteriosclerotic heart disease, elevated cholesterol, Mod |
| Sertraline + Tramadol | known | moderate (type-based estimate (known interaction)) | 0.76 | Tramadol → Seizure ← Sertraline | insomnia, convulsion, intervertebral disc disorder |
| Fluoxetine + Tramadol | known | moderate (type-based estimate (known interaction)) | 0.75 | Tramadol → Seizure ← Fluoxetine | insomnia, convulsion, bruxism |
| Citalopram + Tramadol | known | moderate (type-based estimate (known interaction)) | 0.76 | Tramadol → Seizure ← Citalopram → typical absence seizure | insomnia, convulsion, major depression |
| Warfarin + Omeprazole | known | moderate (DDInter) | 0.82 | Omeprazole → CYP2C18 ← Warfarin | Mumps, arteriosclerotic heart disease, cellulitis |
| Simvastatin + Omeprazole | known | moderate (DDInter) | 0.83 | Omeprazole → CYP2C8 ← Simvastatin | arteriosclerotic heart disease, apoplexy, heart attack |

## Top novel predictions among 38 common drugs

51 of 703 pairs have no recorded interaction in DrugBank, PrimeKG or DDInter. The highest-scoring ones are listed for manual checking against Drugs.com or DrugBank; the last column is for the reviewer.

| Pair | Score | Predicted mechanism | Top path | Reference check |
|---|---|---|---|---|
| Allopurinol + Atorvastatin | 0.83 | The risk or severity of adverse effects can be increased when Allopurinol is combined with Atorvastatin. | Allopurinol → Nephritis ← Atorvastatin | |
| Furosemide + Atorvastatin | 0.80 | The risk or severity of adverse effects can be increased when Furosemide is combined with Atorvastatin. | Furosemide → UGT1A1 ← Atorvastatin | |
| Tramadol + Azithromycin | 0.78 | The metabolism of Azithromycin can be decreased when combined with Tramadol. | Tramadol → CYP3A4 ← Azithromycin | |
| Tramadol + Acetaminophen | 0.77 | The risk or severity of adverse effects can be increased when Tramadol is combined with Acetaminophen. | Tramadol → TRPV1 ← Acetaminophen | |
| Azithromycin + Alprazolam | 0.76 | Azithromycin may increase the QTc-prolonging activities of Alprazolam. | Alprazolam → CYP3A4 ← Azithromycin → arrhythmia | |
| Phenytoin + Lisinopril | 0.73 | The serum concentration of Phenytoin can be decreased when it is combined with Lisinopril. | Phenytoin → Pulmonary infiltrates ← Lisinopril | |
| Lisinopril + Gabapentin | 0.72 | The risk or severity of adverse effects can be increased when Lisinopril is combined with Gabapentin. | Lisinopril → Anuria ← Gabapentin | |
| Acetaminophen + Gabapentin | 0.71 | The risk or severity of adverse effects can be increased when Gabapentin is combined with Acetaminophen. | Acetaminophen → Fulminant hepatitis ← Gabapentin | |
| Digoxin + Sertraline | 0.71 | Sertraline may decrease the cardiotoxic activities of Digoxin. | Digoxin → ABCB1 ← Sertraline | |
| Allopurinol + Losartan | 0.70 | The risk of a hypersensitivity reaction to Losartan is increased when it is combined with Allopurinol. | Allopurinol → Thrombocytopenia ← Losartan | |
| Diltiazem + Gabapentin | 0.69 | The risk or severity of adverse effects can be increased when Gabapentin is combined with Diltiazem. | Diltiazem → Prolonged bleeding time ← Gabapentin | |
| Amlodipine + Gabapentin | 0.69 | The risk or severity of adverse effects can be increased when Gabapentin is combined with Amlodipine. | Amlodipine → CACNA1B ← Gabapentin | |
| Amlodipine + Ciprofloxacin | 0.68 | The risk or severity of adverse effects can be increased when Amlodipine is combined with Ciprofloxacin. | Amlodipine → CYP3A5 ← Ciprofloxacin | |
| Hydrochlorothiazide + Atorvastatin | 0.67 | The risk or severity of adverse effects can be increased when Hydrochlorothiazide is combined with Atorvastatin. | Hydrochlorothiazide → ABCC4 ← Atorvastatin | |
| Furosemide + Sertraline | 0.63 | The therapeutic efficacy of Furosemide can be decreased when used in combination with Sertraline. | Furosemide → ALB ← Sertraline | |