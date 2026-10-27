"""Paths and shared constants for the DDI project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

for _p in (RAW, PROCESSED, ARTIFACTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# Morgan fingerprint settings (baseline features and GNN drug-node features)
FP_RADIUS = 2
FP_BITS = 1024

# Decagon-style filter: keep side effects that occur in at least this many TWOSIDES pairs
MIN_SE_PAIRS = 500

# Severity weights used for the regimen risk score
SEVERITY_ORDER = ["major", "moderate", "minor", "none"]
SEVERITY_WEIGHT = {"major": 1.0, "moderate": 0.6, "minor": 0.3, "none": 0.0}

DISCLAIMER = (
    "Decision support only. This tool is a research prototype, not a medical device. "
    "Predictions come from statistical models trained on public databases and can be wrong or incomplete. "
    "A pair with no flagged interaction has not been shown to be safe; it may simply be untested. "
    "Always confirm with a pharmacist, prescriber, or an authoritative reference before changing any medication."
)
