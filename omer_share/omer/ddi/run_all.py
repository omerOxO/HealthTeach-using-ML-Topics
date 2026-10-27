"""Run the full experiment grid and fill reports/metrics.json.

    python -m ddi.run_all            # everything (about an hour on an RTX 2050)
    python -m ddi.run_all --quick    # R-GCN on merged labels only (what the app serves)

Order: the model the app serves first, then the rest of the grid.
"""
import argparse
import time
import traceback

from . import baseline, gnn

JOBS = [
    ("gnn", "merged", "random", "rgcn"),
    ("gnn", "merged", "cold", "rgcn"),
    ("base", "merged", "random", "lr"),
    ("base", "merged", "random", "xgb"),
    ("base", "merged", "cold", "lr"),
    ("base", "merged", "cold", "xgb"),
    ("gnn", "merged", "random", "rgat"),
    ("gnn", "merged", "cold", "rgat"),
    ("gnn", "merged", "random", "rgcn_gin"),  # R-GCN with GIN molecular-graph drug features
    ("gnn", "merged", "cold", "rgcn_gin"),
    ("base", "tdc", "random", "lr"),
    ("base", "tdc", "random", "xgb"),
    ("base", "tdc", "cold", "lr"),
    ("base", "tdc", "cold", "xgb"),
    ("gnn", "tdc", "random", "rgcn"),
    ("gnn", "tdc", "cold", "rgcn"),
    ("gnn", "tdc", "random", "rgat"),
    ("gnn", "tdc", "cold", "rgat"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    jobs = JOBS[:2] if args.quick else JOBS
    failed = []
    for kind, labels, split, model in jobs:
        t0 = time.time()
        print(f"\n######## {kind} {model} labels={labels} split={split}", flush=True)
        try:
            if kind == "gnn":
                conv, _, enc = model.partition("_")
                gnn.run(labels, split, conv, drug_encoder=enc or "fp")
            else:
                baseline.run(labels, split, model, 150_000)
        except Exception:
            traceback.print_exc()
            failed.append((kind, labels, split, model))
        print(f"######## done in {time.time() - t0:.0f}s", flush=True)
    print("\nFAILED:" if failed else "\nall jobs finished", failed or "")


if __name__ == "__main__":
    main()
