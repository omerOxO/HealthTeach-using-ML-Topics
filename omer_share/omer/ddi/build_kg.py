"""Build the drug knowledge graph from the raw sources.

    python -m ddi.build_kg

1. TDC DrugBank DDI   -> drug universe (DrugBank IDs + SMILES) and 86 typed interaction labels
2. PrimeKG            -> drug names, targets/enzymes/transporters/carriers, SIDER side effects,
                         indications, contraindications, off-label uses, ATC classes, extra DrugBank DDIs
3. TWOSIDES           -> pair-level polypharmacy side effects (PubChem CIDs mapped by InChIKey)
4. DDInter            -> DDIs with Major/Moderate/Minor severity (mapped by drug name)
5. networkx graph     -> saved as kg.pkl, then converted to a PyG HeteroData (hetero.pt)

Drug-drug interactions are *labels*, not message-passing edges, so they are stored separately in
ddi.parquet. Training scripts decide which DDI edges the GNN may see for each split.
"""
import pickle
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import torch
from torch_geometric.data import HeteroData

from .chem import inchikey_block, morgan
from .config import MIN_SE_PAIRS, PROCESSED, RAW

SEV_RANK = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
PRIMEKG_RELS = ["drug_protein", "drug_effect", "indication", "contraindication",
                "off-label use", "drug_drug", "protein_protein"]
# PrimeKG relation -> our edge relation name (drug_protein is split by display_relation)
DISEASE_RELS = {"indication": "indication", "contraindication": "contraindication",
                "off-label use": "off_label"}


def log(msg):
    print(f"[build_kg] {msg}", flush=True)


# ---------------------------------------------------------------- 1. TDC DrugBank DDI
def load_tdc():
    tdc = pd.read_csv(RAW / "tdc/drugbank.tab", sep="\t")
    a = tdc[["ID1", "X1"]].set_axis(["drugbank_id", "smiles"], axis=1)
    b = tdc[["ID2", "X2"]].set_axis(["drugbank_id", "smiles"], axis=1)
    drugs = pd.concat([a, b]).drop_duplicates("drugbank_id").sort_values("drugbank_id")
    drugs = drugs.reset_index(drop=True)
    drugs.insert(0, "drug_idx", np.arange(len(drugs)))
    types = (tdc[["Y", "Map"]].drop_duplicates("Y").sort_values("Y")
             .rename(columns={"Y": "type_id", "Map": "template"}).reset_index(drop=True))
    log(f"TDC: {len(tdc):,} typed pairs, {len(drugs):,} drugs, {len(types)} interaction types")
    return tdc, drugs, types


# ---------------------------------------------------------------- 2. PrimeKG
def load_primekg():
    cols = ["relation", "display_relation", "x_index", "x_id", "x_type", "x_name",
            "y_id", "y_type", "y_name"]
    tbl = pacsv.read_csv(
        RAW / "primekg/kg.csv",
        convert_options=pacsv.ConvertOptions(include_columns=cols,
                                             column_types={c: pa.string() for c in cols}),
    )
    tbl = tbl.filter(pc.is_in(tbl["relation"], value_set=pa.array(PRIMEKG_RELS)))
    df = tbl.to_pandas()
    log(f"PrimeKG: {len(df):,} rows in the relations we use")
    return df


def parse_after(text, marker):
    if not isinstance(text, str) or marker not in text:
        return None
    out = text.split(marker, 1)[1].strip().rstrip(".").strip()
    return out or None


def split_classes(text):
    """PrimeKG joins a drug's several ATC classes with ' and ' -> '; '-separated list.
    A few genuine class names contain ' and ' and get split too; acceptable for a class node."""
    if not text:
        return None
    parts = [p.strip(" .,") for p in text.split(" and ")]
    parts = list(dict.fromkeys(p for p in parts if len(p) > 3))
    return "; ".join(parts) or None


def load_drug_features():
    feats = pd.read_csv(RAW / "primekg/drug_features.tab", sep="\t", dtype=str)
    feats["atc3"] = feats["atc_3"].map(lambda t: split_classes(parse_after(t, "pharmacologically related to ")))
    # "The chemical and functional group of  is retinoids for treatment of psoriasis."
    feats["atc4"] = feats["atc_4"].map(
        lambda t: split_classes(t.rsplit(" is ", 1)[1].strip().rstrip("."))
        if isinstance(t, str) and " is " in t else None)
    feats["categories"] = feats["category"].map(
        lambda t: "; ".join(c.strip().rstrip(".") for c in parse_after(t, " is part of ").split(" ; "))
        if parse_after(t, " is part of ") else None)
    feats["group"] = feats["group"].map(lambda t: parse_after(t, " is "))
    keep = ["node_index", "atc3", "atc4", "categories", "group", "description", "mechanism_of_action",
            "indication", "half_life"]
    return feats[keep]


# ---------------------------------------------------------------- 3. TWOSIDES
def load_twosides(drugs):
    cid_smiles, parts = {}, []
    for chunk in pd.read_csv(RAW / "tdc/twosides.csv", chunksize=500_000,
                             usecols=["ID1", "ID2", "Y", "Side Effect Name", "X1", "X2"]):
        for idc, sc in (("ID1", "X1"), ("ID2", "X2")):
            for cid, smi in chunk[[idc, sc]].drop_duplicates(idc).itertuples(index=False):
                cid_smiles.setdefault(cid, smi)
        parts.append(chunk[["ID1", "ID2", "Y", "Side Effect Name"]])
    tw = pd.concat(parts, ignore_index=True)
    log(f"TWOSIDES: {len(tw):,} (pair, side effect) rows, {len(cid_smiles)} drugs")

    key2idx = {}
    for idx, k in zip(drugs["drug_idx"], drugs["inchikey"]):
        if k:
            key2idx.setdefault(k, idx)
    cid2idx = {cid: key2idx.get(inchikey_block(s)) for cid, s in cid_smiles.items()}
    mapped = sum(v is not None for v in cid2idx.values())
    log(f"TWOSIDES: mapped {mapped}/{len(cid2idx)} drugs to DrugBank by InChIKey")

    tw["a"] = tw["ID1"].map(cid2idx)
    tw["b"] = tw["ID2"].map(cid2idx)
    tw = tw.dropna(subset=["a", "b"])
    tw = tw[tw["a"] != tw["b"]]
    a, b = tw["a"].astype(int).to_numpy(), tw["b"].astype(int).to_numpy()
    tw = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b),
                       "se_raw": tw["Y"].to_numpy(), "se_name": tw["Side Effect Name"].to_numpy()})
    tw = tw.drop_duplicates(["a", "b", "se_raw"])

    counts = tw.groupby("se_raw")["a"].size()
    keep = counts[counts >= MIN_SE_PAIRS].index
    tw = tw[tw["se_raw"].isin(keep)]
    ses = (tw.groupby(["se_raw", "se_name"]).size().rename("n_pairs").reset_index()
           .sort_values("n_pairs", ascending=False).reset_index(drop=True))
    ses.insert(0, "se_idx", np.arange(len(ses)))
    tw = tw.merge(ses[["se_raw", "se_idx"]], on="se_raw")[["a", "b", "se_idx"]]
    log(f"TWOSIDES: {tw[['a','b']].drop_duplicates().shape[0]:,} mapped pairs, "
        f"{len(ses)} side effects with >= {MIN_SE_PAIRS} pairs")
    return tw.reset_index(drop=True), ses.drop(columns="se_raw")


# ---------------------------------------------------------------- 4. DDInter
def load_ddinter(name2idx):
    files = sorted((RAW / "ddinter").glob("code_*.csv"))
    if not files:
        log("DDInter: no files, severity falls back to interaction-type rules")
        return pd.DataFrame(columns=["a", "b", "ddinter_level"])
    dd = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dd["a"] = dd["Drug_A"].str.lower().map(name2idx)
    dd["b"] = dd["Drug_B"].str.lower().map(name2idx)
    n_all = len(dd)
    dd = dd.dropna(subset=["a", "b"])
    a, b = dd["a"].astype(int).to_numpy(), dd["b"].astype(int).to_numpy()
    dd = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b),
                       "rank": dd["Level"].map(SEV_RANK).fillna(0).astype(int).to_numpy()})
    dd = dd[dd["a"] != dd["b"]].groupby(["a", "b"], as_index=False)["rank"].max()
    inv = {v: k for k, v in SEV_RANK.items()}
    dd["ddinter_level"] = dd.pop("rank").map(inv)
    log(f"DDInter: {n_all:,} rows from {len(files)} files -> {len(dd):,} pairs inside the drug universe")
    return dd


# ---------------------------------------------------------------- main
def main():
    tdc, drugs, types = load_tdc()
    drugs["inchikey"] = [inchikey_block(s) for s in drugs["smiles"]]
    db2idx = dict(zip(drugs["drugbank_id"], drugs["drug_idx"]))

    pk = load_primekg()
    drug_rows = pk[(pk["x_type"] == "drug") & pk["x_id"].isin(db2idx)]

    # names and PrimeKG features
    names = drug_rows.drop_duplicates("x_id").set_index("x_id")
    drugs["name"] = drugs["drugbank_id"].map(names["x_name"]).fillna(drugs["drugbank_id"])
    drugs["primekg_index"] = drugs["drugbank_id"].map(names["x_index"])
    feats = load_drug_features()
    drugs = drugs.merge(feats, how="left", left_on="primekg_index", right_on="node_index").drop(columns="node_index")
    drugs["in_primekg"] = drugs["primekg_index"].notna()
    log(f"{drugs['in_primekg'].sum()}/{len(drugs)} TDC drugs found in PrimeKG")

    # ---- knowledge-graph nodes and edges (everything except DDIs)
    nodes = [("drug", f"drug:{r.drugbank_id}", r.name, r.drug_idx) for r in drugs.itertuples()]
    node_id = {k: i for i, (_, k, _, _) in enumerate(nodes)}
    edges = []  # (src_key, rel, dst_key)

    def add_node(ntype, key, name):
        if key not in node_id:
            node_id[key] = len(nodes)
            nodes.append((ntype, key, name, None))

    dp = drug_rows[drug_rows["relation"] == "drug_protein"]
    for r in dp.itertuples():
        add_node("protein", f"protein:{r.y_id}", r.y_name)
        edges.append((f"drug:{r.x_id}", r.display_relation, f"protein:{r.y_id}"))
    de = drug_rows[drug_rows["relation"] == "drug_effect"]
    for r in de.itertuples():
        add_node("effect", f"effect:{r.y_id}", r.y_name)
        edges.append((f"drug:{r.x_id}", "side_effect", f"effect:{r.y_id}"))
    dz = drug_rows[drug_rows["relation"].isin(DISEASE_RELS)]
    for r in dz.itertuples():
        add_node("disease", f"disease:{r.y_id}", r.y_name)
        edges.append((f"drug:{r.x_id}", DISEASE_RELS[r.relation], f"disease:{r.y_id}"))
    for r in drugs.itertuples():
        for level in ("atc3", "atc4"):
            classes = getattr(r, level)
            if isinstance(classes, str):
                for cls in classes.split("; "):
                    key = f"atc:{level}:{cls.lower()}"
                    add_node("atc", key, cls[0].upper() + cls[1:])
                    edges.append((f"drug:{r.drugbank_id}", "in_class", key))
    pp = pk[pk["relation"] == "protein_protein"]
    pp = pp[pp["x_id"].radd("protein:").isin(node_id) & pp["y_id"].radd("protein:").isin(node_id)]
    pp = pp[pp["x_id"] < pp["y_id"]]  # PrimeKG lists both directions
    for r in pp.itertuples():
        edges.append((f"protein:{r.x_id}", "ppi", f"protein:{r.y_id}"))
    edges = sorted(set(edges))

    nodes_df = pd.DataFrame(nodes, columns=["ntype", "key", "name", "drug_idx"])
    nodes_df.insert(0, "node_id", np.arange(len(nodes_df)))
    nodes_df["local_idx"] = nodes_df.groupby("ntype").cumcount()
    edges_df = pd.DataFrame(edges, columns=["src_key", "rel", "dst_key"])
    edges_df["src"] = edges_df["src_key"].map(node_id)
    edges_df["dst"] = edges_df["dst_key"].map(node_id)
    edges_df = edges_df[["src", "rel", "dst"]]
    log("KG nodes: " + ", ".join(f"{k}={v:,}" for k, v in nodes_df["ntype"].value_counts().items()))
    log("KG edges: " + ", ".join(f"{k}={v:,}" for k, v in edges_df["rel"].value_counts().items()))

    # per-drug counts for EDA and the cold-start discussion
    drug_edges = edges_df[edges_df["src"] < len(drugs)]
    cnt = drug_edges.groupby(["src", "rel"]).size().unstack(fill_value=0)
    for rel in ["target", "enzyme", "transporter", "carrier", "side_effect", "indication", "in_class"]:
        drugs[f"n_{rel}"] = drugs["drug_idx"].map(cnt[rel] if rel in cnt else {}).fillna(0).astype(int)

    # ---- DDI labels: TDC (typed) + PrimeKG drug_drug + DDInter
    a, b = tdc["ID1"].map(db2idx).to_numpy(), tdc["ID2"].map(db2idx).to_numpy()
    t = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b), "tdc_type": tdc["Y"].to_numpy(),
                      "tdc_drug1": a})
    t = t.drop_duplicates(["a", "b"])  # a pair occasionally has two types; keep the first
    t["in_tdc"] = True

    dd = drug_rows[(drug_rows["relation"] == "drug_drug") & drug_rows["y_id"].isin(db2idx)]
    a, b = dd["x_id"].map(db2idx).to_numpy(), dd["y_id"].map(db2idx).to_numpy()
    p = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b)}).drop_duplicates()
    p = p[p["a"] != p["b"]]
    p["in_primekg"] = True

    name2idx = {}
    for idx, n in zip(drugs["drug_idx"], drugs["name"]):
        name2idx.setdefault(str(n).lower(), idx)
    ddi_dd = load_ddinter(name2idx)
    ddi_dd["in_ddinter"] = True

    ddi = t.merge(p, on=["a", "b"], how="outer").merge(ddi_dd, on=["a", "b"], how="outer")
    for c in ("in_tdc", "in_primekg", "in_ddinter"):
        ddi[c] = ddi[c].fillna(False).astype(bool)
    ddi["tdc_type"] = ddi["tdc_type"].fillna(-1).astype(int)
    ddi["tdc_drug1"] = ddi["tdc_drug1"].fillna(-1).astype(int)
    ddi = ddi.sort_values(["a", "b"]).reset_index(drop=True)
    log(f"DDI labels: tdc={ddi['in_tdc'].sum():,}  primekg={ddi['in_primekg'].sum():,}  "
        f"ddinter={ddi['in_ddinter'].sum():,}  union={len(ddi):,}")

    # ---- TWOSIDES
    tw, ses = load_twosides(drugs)
    drugs["in_twosides"] = drugs["drug_idx"].isin(set(tw["a"]) | set(tw["b"]))

    # ---- fingerprints
    fps = morgan(drugs["smiles"].tolist())
    drugs["fp_ok"] = fps.sum(1) > 0
    log(f"Morgan fingerprints: {drugs['fp_ok'].sum()}/{len(drugs)} drugs parsed")

    # ---- networkx graph (used for path explanations and the graph explorer)
    G = nx.MultiGraph()
    for r in nodes_df.itertuples():
        G.add_node(r.node_id, ntype=r.ntype, name=r.name, key=r.key)
    G.add_edges_from((s, d, {"rel": rel}) for s, rel, d in edges_df.itertuples(index=False))
    log(f"networkx: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ---- PyG HeteroData (converted from the networkx graph)
    data = HeteroData()
    for ntype, grp in nodes_df.groupby("ntype"):
        data[ntype].num_nodes = len(grp)
    data["drug"].x = torch.from_numpy(fps).float()
    local = nodes_df["local_idx"].to_numpy()
    ntypes = nodes_df["ntype"].to_numpy()
    by_rel = {}
    for s, d, attr in G.edges(data=True):
        by_rel.setdefault((ntypes[s], attr["rel"], ntypes[d]), []).append((local[s], local[d]))
    for (st, rel, dt), pairs in by_rel.items():
        data[st, rel, dt].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    log(f"HeteroData edge types: {len(data.edge_types)}")

    # ---- save
    drugs.to_parquet(PROCESSED / "drugs.parquet", index=False)
    nodes_df.to_parquet(PROCESSED / "nodes.parquet", index=False)
    edges_df.to_parquet(PROCESSED / "edges.parquet", index=False)
    ddi.to_parquet(PROCESSED / "ddi.parquet", index=False)
    types.to_parquet(PROCESSED / "ddi_types.parquet", index=False)
    tw.to_parquet(PROCESSED / "twosides.parquet", index=False)
    ses.to_parquet(PROCESSED / "side_effects.parquet", index=False)
    np.save(PROCESSED / "fingerprints.npy", fps)
    with open(PROCESSED / "kg.pkl", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(data, PROCESSED / "hetero.pt")
    log(f"saved to {PROCESSED}")


if __name__ == "__main__":
    main()
