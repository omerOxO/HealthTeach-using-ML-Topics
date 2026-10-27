"""Do the model's attributions agree with the knowledge-graph path explanations?

    python -m ddi.explain_eval

For each pair we compare two rankings of the drugs' neighbour nodes:
- path nodes: the nodes on the ranked KG paths of ddi/explain.py (shared enzyme, target, class ...,
  protein-protein bridges, the effect tail), in path order;
- model nodes: the GNNExplainer node ranking of ddi/model_explain.py for the served R-GCN.

Measures (k = 1, 3, 5):
- overlap@k:  |model top-k ∩ path top-k| / k;
- path-top in model@k: is the best path's node among the model's top-k?
- model-top on a path: is the model's top node anywhere on the path explanation?
- random baseline for overlap@k: what k neighbours drawn at random from the candidate set would score.
Also fidelity: the score keeping only the top-8 neighbours' links, and with all links but theirs.

Pairs: the case studies of ddi/case_studies.py, plus a random sample of known test pairs from the
merged/random split (the sample says how typical the case studies are).
Writes reports/explanation_agreement.md.
"""
import argparse
import time

import numpy as np
import pandas as pd

from .case_studies import CASES
from .config import REPORTS, SEED
from .service import DDIService
from .splits import load_positive_pairs, load_split

KS = (1, 3, 5)


def path_nodes(ex):
    out = []
    for p in ex["paths"]:
        for v in p["via"]:
            if v["id"] not in out:
                out.append(v["id"])
    tail = ex["effect_tail"]
    if tail and tail["node"] not in out:
        out.append(tail["node"])
    return out


def compare(s, a, b):
    t0 = time.time()
    m = s.model_explanation(a, b)
    secs = time.time() - t0
    row = s._pair(a, b)
    ex = s.explainer.explain(a, b, [e["name"] for e in row["effects"][:3]],
                             mechanism=row["mechanism"] if row["status"] != "none" else None)
    P = path_nodes(ex)
    M = [n["id"] for n in m["top_nodes"]]
    n_cand = len(set(s.G[a]) | set(s.G[b]) | set(s.partners.get(a, [])) | set(s.partners.get(b, []))) - 2
    out = {"pair": f"{s.names[a]} + {s.names[b]}", "status": row["status"], "score": m["score"],
           "n_path_nodes": len(P), "n_cand": n_cand,
           "model_top": m["top_nodes"][0]["name"] if M else "–",
           "path_top": ex["paths"][0]["via"][0]["name"] if ex["paths"] else "–",
           "model_top_on_path": bool(M and M[0] in P), "score_only_top": m["score_only_top"],
           "score_without_top": m["score_without_top"], "seconds": secs}
    for k in KS:
        out[f"overlap@{k}"] = len(set(M[:k]) & set(P[:k])) / k if P else np.nan
        out[f"path_top_in_model@{k}"] = float(P[0] in M[:k]) if P else np.nan
        out[f"random@{k}"] = min(len(P), k) / max(n_cand, 1) if P else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200, help="random known test pairs to add")
    args = ap.parse_args()
    s = DDIService()
    case_rows = []
    for x, y in CASES:
        a, b = s.resolve(x), s.resolve(y)
        if a is not None and b is not None:
            case_rows.append(compare(s, a, b))

    ddi = load_positive_pairs("merged")
    split = load_split("merged", "random")
    rng = np.random.default_rng(SEED)
    pick = rng.choice(split["test_pos"], args.sample, replace=False)
    sample_rows = [compare(s, int(r.a), int(r.b)) for r in ddi.iloc[pick].itertuples(index=False)]
    cases, sample = pd.DataFrame(case_rows), pd.DataFrame(sample_rows)

    def agg(df):
        with_paths = df[df["n_path_nodes"] > 0]
        r = {"pairs": len(df), "with a KG path": len(with_paths)}
        for k in KS:
            r[f"overlap@{k}"] = with_paths[f"overlap@{k}"].mean()
            r[f"random@{k}"] = with_paths[f"random@{k}"].mean()
        for k in KS:
            r[f"path top in model top-{k}"] = with_paths[f"path_top_in_model@{k}"].mean()
        r["model top on a path"] = with_paths["model_top_on_path"].mean()
        r["score, all links"] = df["score"].mean()
        r["score, only top-8 links"] = df["score_only_top"].mean()
        r["score, all but top-8"] = df["score_without_top"].mean()
        r["seconds per explanation"] = df["seconds"].mean()
        return r

    A, B = agg(cases), agg(sample)
    md = ["# Model attributions vs knowledge-graph path explanations", "",
          f"Model: {s.model_info['file']}. Model attributions come from GNNExplainer (`ddi/model_explain.py`): a "
          "learned soft mask over the edges feeding the two drugs' embeddings (knowledge-graph links and known-"
          "interaction links), ranked into neighbour nodes. Path explanations come from `ddi/explain.py`: shared "
          "enzymes, targets, transporters, classes, side effects and protein-protein bridges, ranked by relation "
          "weight and specificity. Both rank neighbour nodes of the two drugs, so we compare the rankings.", "",
          "## Summary", "",
          f"| Measure | Case studies ({A['pairs']} pairs) | Random known test pairs ({B['pairs']}) |", "|---|---|---|"]
    for key in A:
        fa, fb = A[key], B[key]
        fmt = (lambda v: f"{v:.0f}") if key in ("pairs", "with a KG path") else \
            (lambda v: f"{v:.3f}") if key.startswith("random") else (lambda v: f"{v:.2f}")
        md.append(f"| {key} | {fmt(fa)} | {fmt(fb)} |")
    md += ["", "`overlap@k` is the share of the model's top-k neighbours that are also among the top-k path nodes; "
           "`random@k` is the overlap that k neighbours picked at random from all neighbours of the two drugs "
           "would get. The score rows are mean predicted probabilities: with only the top-8 neighbours' links "
           "kept, and with every link except theirs.", "",
           "## Case studies", "",
           "| Pair | Status | Score | Model's top neighbour | Best KG path node | Overlap@3 | Overlap@5 | "
           "Model top on a path | Score, only top-8 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in case_rows:
        md.append(f"| {r['pair']} | {r['status']} | {r['score']:.2f} | {r['model_top']} | {r['path_top']} | "
                  f"{r['overlap@3']:.2f} | {r['overlap@5']:.2f} | {'yes' if r['model_top_on_path'] else 'no'} | "
                  f"{r['score_only_top']:.2f} |")
    md += ["", "## Reading the result", "", *interpret(A, B)]
    (REPORTS / "explanation_agreement.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def interpret(A, B):
    lines = []
    for name, r in (("case studies", A), ("random test pairs", B)):
        lift = r["overlap@5"] / r["random@5"] if r["random@5"] else float("nan")
        lines.append(f"- On the {name}, the model's top-5 neighbours overlap the top-5 path nodes "
                     f"{r['overlap@5']:.0%} of the time, {lift:.0f}x what random neighbours would give; the "
                     f"model's top neighbour lies on a path explanation in {r['model top on a path']:.0%} of pairs.")
    lines.append(f"- Keeping only the top-8 neighbours' links keeps the mean score at "
                 f"{B['score, only top-8 links']:.2f} (vs {B['score, all links']:.2f} with every link), so the "
                 "attribution captures what the model uses. Removing them lowers it only to "
                 f"{B['score, all but top-8']:.2f}: the remaining links carry much of the same signal (many "
                 "neighbours are redundant), so the attributions are sufficient rather than necessary.")
    lines.append("- Disagreements are expected: the path explainer only looks at neighbours the two drugs share "
                 "(or bridge through a protein link), while the model can also lean on one drug's own neighbours "
                 "(its class, indications or a single target) that tell it the drug is interaction-prone. Those "
                 "show up in the app as neighbours of one drug only, without the \"also a KG path\" tag.")
    return lines


if __name__ == "__main__":
    main()
