# EDA summary

- Drug universe: **1,706** DrugBank drugs with SMILES (from the TDC DrugBank DDI benchmark).
- KG nodes: disease 1,903, drug 1,706, protein 1,588, effect 970, atc 801
- KG edges: side_effect 58,639, contraindication 26,067, indication 7,291, ppi 6,793, target 6,692, in_class 5,392, enzyme 3,809, off_label 2,307, transporter 2,082, carrier 542
- Known interacting pairs: TDC 191,402, PrimeKG 457,564, DDInter 94,569, merged **526,105** of 1,454,365 possible pairs (density 36.2%; TDC alone 13.2%).
- Class imbalance: about 1 known pair per 2.8 possible pairs; unknown pairs are *unlabelled*, not safe.
- Most connected drugs: Quinidine (1,388), Cyclosporine (1,382), Chlorpromazine (1,358), Carbamazepine (1,342), Clozapine (1,341), Haloperidol (1,334), Amitriptyline (1,333), Imipramine (1,324)
- Cold-start exposure: 183 drugs have no target/enzyme/transporter link, 723 have no SIDER side effect, 33 have no KG edge at all, 0 have no known interaction.
- TWOSIDES: 50,150 mapped pairs over 566 drugs, 915 side effects with >=500 pairs; median side effects per pair 53; median side-effect positive rate 5.08% (strong per-label imbalance).
- Most frequent TWOSIDES effects: arterial pressure NOS decreased, anaemia, Difficulty breathing, nausea, neumonia, Fatigue, Pain, diarrhea, asthenia, emesis

## Source overlap (merged DDI labels)

- PrimeKG: 277,463
- TDC+PrimeKG: 116,207
- TDC: 37,866
- TDC+PrimeKG+DDInter: 33,554
- PrimeKG+DDInter: 30,340
- DDInter: 26,900
- TDC+DDInter: 3,775