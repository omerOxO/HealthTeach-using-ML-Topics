"""Download every raw source used by the pipeline.

    python -m ddi.download            # everything
    python -m ddi.download --only tdc # one group

Files land in data/raw/<group>/. Existing complete files are skipped.
"""
import argparse
import sys

import requests
from tqdm import tqdm

from .config import RAW

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{}"
DDINTER = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{}.csv"
# DDInter splits its downloads by the ATC first letter
DDINTER_CODES = ["A", "B", "D", "H", "L", "P", "R", "V"]

SOURCES = {
    "tdc": {
        # TDC multi_pred DDI benchmarks (DrugBank DDI from DeepDDI, TWOSIDES from Decagon)
        "drugbank.tab": DATAVERSE.format(4139573),
        "twosides.csv": DATAVERSE.format(4139574),
    },
    "primekg": {
        # PrimeKG (Chandak et al. 2023), doi:10.7910/DVN/IXA7BM
        "kg.csv": DATAVERSE.format(6180620),
        "drug_features.tab": DATAVERSE.format(6180619),
    },
    "ddinter": {f"code_{c}.csv": DDINTER.format(c) for c in DDINTER_CODES},
}


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SOURCES))
    args = ap.parse_args()
    failed = []
    for group, files in SOURCES.items():
        if args.only and group != args.only:
            continue
        (RAW / group).mkdir(parents=True, exist_ok=True)
        for name, url in files.items():
            dest = RAW / group / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"skip {group}/{name} (exists)")
                continue
            try:
                fetch(url, dest)
            except Exception as e:  # DDInter in particular is often slow or down
                print(f"FAILED {group}/{name}: {e}", file=sys.stderr)
                failed.append(f"{group}/{name}")
    if failed:
        print("Some downloads failed; the KG build skips missing optional sources (DDInter):", failed)


if __name__ == "__main__":
    main()
