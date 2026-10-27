"""Exploratory data analysis of the built KG.

    python -m ddi.eda

Writes reports/eda.json (consumed by the Analytics page) and reports/eda.md (readable summary).
"""
import json

import numpy as np
import pandas as pd

from .config import PROCESSED, REPORTS


def log_hist(values, n_bins=16):
    values = np.asarray(values)
    hi = max(1, int(values.max()))
    edges = np.unique(np.round(np.logspace(0, np.log10(hi + 1), n_bins)).astype(int))
    counts = [int(((values >= lo) & (values < up)).sum()) for lo, up in zip(edges[:-1], edges[1:])]
    return [{"lo": int(lo), "hi": int(up) - 1, "count": c} for lo, up, c in zip(edges[:-1], edges[1:], counts)]


def main():
    drugs = pd.read_parquet(PROCESSED / "drugs.parquet")
    nodes = pd.read_parquet(PROCESSED / "nodes.parquet")
    edges = pd.read_parquet(PROCESSED / "edges.parquet")
    ddi = pd.read_parquet(PROCESSED / "ddi.parquet")
    types = pd.read_parquet(PROCESSED / "ddi_types.parquet")
    tw = pd.read_parquet(PROCESSED / "twosides.parquet")
    ses = pd.read_parquet(PROCESSED / "side_effects.parquet")
    n = len(drugs)
    possible = n * (n - 1) // 2

    deg = np.bincount(np.r_[ddi["a"], ddi["b"]], minlength=n)
    deg_tdc = np.bincount(np.r_[ddi.loc[ddi.in_tdc, "a"], ddi.loc[ddi.in_tdc, "b"]], minlength=n)
    top = np.argsort(-deg)[:20]

    src = ddi[["in_tdc", "in_primekg", "in_ddinter"]]
    overlap = (src.astype(int).astype(str).agg("".join, axis=1).value_counts()
               .rename(lambda k: "+".join(n for n, f in zip(["TDC", "PrimeKG", "DDInter"], k) if f == "1")))

    no_targets = int((drugs[["n_target", "n_enzyme", "n_transporter", "n_carrier"]].sum(axis=1) == 0).sum())
    no_se = int((drugs["n_side_effect"] == 0).sum())
    isolated = int(((drugs[["n_target", "n_enzyme", "n_transporter", "n_carrier", "n_side_effect",
                            "n_indication", "n_in_class"]].sum(axis=1)) == 0).sum())

    type_counts = (ddi[ddi.tdc_type >= 0].groupby("tdc_type").size().rename("count").reset_index()
                   .merge(types, left_on="tdc_type", right_on="type_id").sort_values("count", ascending=False))
    tw_pairs = tw[["a", "b"]].drop_duplicates()
    se_per_pair = tw.groupby(["a", "b"]).size()

    out = {
        "graph": {
            "nodes": {k: int(v) for k, v in nodes["ntype"].value_counts().items()},
            "edges": {k: int(v) for k, v in edges["rel"].value_counts().items()},
        },
        "ddi": {
            "n_drugs": n, "possible_pairs": possible,
            "positives": {"tdc": int(ddi.in_tdc.sum()), "primekg": int(ddi.in_primekg.sum()),
                          "ddinter": int(ddi.in_ddinter.sum()), "merged": len(ddi)},
            "density": {"tdc": float(ddi.in_tdc.sum() / possible), "merged": float(len(ddi) / possible)},
            "source_overlap": {k: int(v) for k, v in overlap.items()},
            "degree_hist": log_hist(deg[deg > 0]),
            "degree_hist_tdc": log_hist(deg_tdc[deg_tdc > 0]),
            "drugs_without_interactions": int((deg == 0).sum()),
            "top_drugs": [{"name": drugs["name"].iloc[i], "id": drugs["drugbank_id"].iloc[i], "degree": int(deg[i])}
                          for i in top],
            "type_counts": [{"type_id": int(r.type_id), "template": r.template, "count": int(r.count)}
                            for r in type_counts.itertuples()],
            "ddinter_levels": {k: int(v) for k, v in ddi["ddinter_level"].value_counts().items()},
        },
        "cold_start": {
            "drugs_without_protein_links": no_targets, "drugs_without_side_effects": no_se,
            "drugs_with_no_kg_edges": isolated, "drugs_not_in_primekg": int((~drugs.in_primekg).sum()),
        },
        "twosides": {
            "pairs": int(len(tw_pairs)), "drugs": int(drugs.in_twosides.sum()), "side_effects": int(len(ses)),
            "rows": int(len(tw)),
            "top_side_effects": [{"name": r.se_name, "pairs": int(r.n_pairs)} for r in ses.head(30).itertuples()],
            "side_effect_freq_hist": log_hist(ses["n_pairs"].to_numpy()),
            "positive_rate_median": float(ses["n_pairs"].median() / len(tw_pairs)),
            "side_effects_per_pair_median": float(se_per_pair.median()),
        },
    }
    (REPORTS / "eda.json").write_text(json.dumps(out, indent=2))

    d, c, t = out["ddi"], out["cold_start"], out["twosides"]
    md = [
        "# EDA summary", "",
        f"- Drug universe: **{n:,}** DrugBank drugs with SMILES (from the TDC DrugBank DDI benchmark).",
        f"- KG nodes: " + ", ".join(f"{k} {v:,}" for k, v in out["graph"]["nodes"].items()),
        f"- KG edges: " + ", ".join(f"{k} {v:,}" for k, v in out["graph"]["edges"].items()),
        f"- Known interacting pairs: TDC {d['positives']['tdc']:,}, PrimeKG {d['positives']['primekg']:,}, "
        f"DDInter {d['positives']['ddinter']:,}, merged **{d['positives']['merged']:,}** of {possible:,} possible "
        f"pairs (density {d['density']['merged']:.1%}; TDC alone {d['density']['tdc']:.1%}).",
        f"- Class imbalance: about 1 known pair per {possible / max(1, d['positives']['merged']):.1f} possible pairs; "
        f"unknown pairs are *unlabelled*, not safe.",
        f"- Most connected drugs: " + ", ".join(f"{x['name']} ({x['degree']:,})" for x in d["top_drugs"][:8]),
        f"- Cold-start exposure: {c['drugs_without_protein_links']} drugs have no target/enzyme/transporter link, "
        f"{c['drugs_without_side_effects']} have no SIDER side effect, {c['drugs_with_no_kg_edges']} have no KG edge at all, "
        f"{d['drugs_without_interactions']} have no known interaction.",
        f"- TWOSIDES: {t['pairs']:,} mapped pairs over {t['drugs']} drugs, {t['side_effects']} side effects with >=500 pairs; "
        f"median side effects per pair {t['side_effects_per_pair_median']:.0f}; median side-effect positive rate "
        f"{t['positive_rate_median']:.2%} (strong per-label imbalance).",
        f"- Most frequent TWOSIDES effects: " + ", ".join(x["name"] for x in t["top_side_effects"][:10]),
        "", "## Source overlap (merged DDI labels)", "",
        *[f"- {k}: {v:,}" for k, v in d["source_overlap"].items()],
    ]
    (REPORTS / "eda.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
