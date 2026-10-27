"""Evaluation metrics and the shared results file (reports/metrics.json)."""
import json

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .config import REPORTS

HITS_K = (20, 50, 100)  # scores should be logits/margins: saturated probabilities create ties
METRICS_FILE = REPORTS / "metrics.json"


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def hits_at_k(pos, neg, ks=HITS_K):
    """OGB-style Hits@K: share of positives scored above the K-th best unlabelled pair."""
    neg_sorted = np.sort(neg)[::-1]
    return {f"hits@{k}": float((pos > neg_sorted[k - 1]).mean()) if len(neg_sorted) >= k else float("nan")
            for k in ks}


def binary_report(pos, neg, threshold):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "f1": float(f1_score(y, s >= threshold)),
        "threshold": float(threshold),
        "n_pos": int(len(pos)),
        "n_unlabelled": int(len(neg)),
    }
    out.update(hits_at_k(pos, neg))
    return out


def evaluate_split(score_fn, pairs, split):
    """score_fn(a, b) -> scores. Threshold for F1 is tuned on validation (uniform negatives)."""
    def sc(idx=None, arr=None):
        if arr is not None:
            return score_fn(arr[:, 0], arr[:, 1])
        return score_fn(pairs[idx, 0], pairs[idx, 1])

    val_pos, val_neg = sc(split["val_pos"]), sc(arr=split["val_neg"])
    thr = best_f1_threshold(np.r_[np.ones(len(val_pos)), np.zeros(len(val_neg))], np.r_[val_pos, val_neg])
    test_pos = sc(split["test_pos"])
    return {
        "val": binary_report(val_pos, val_neg, thr),
        "test": binary_report(test_pos, sc(arr=split["test_neg"]), thr),
        "test_hard": binary_report(test_pos, sc(arr=split["test_neg_hard"]), thr),
    }


def per_label_auprc(y, s, min_pos=5):
    """Mean per-side-effect AUPRC and AUROC (labels with fewer than min_pos positives skipped)."""
    aps, aucs, kept = [], [], []
    for j in range(y.shape[1]):
        pos = y[:, j].sum()
        if pos < min_pos or pos == len(y):
            continue
        aps.append(average_precision_score(y[:, j], s[:, j]))
        aucs.append(roc_auc_score(y[:, j], s[:, j]))
        kept.append(j)
    return {"mean_auprc": float(np.mean(aps)), "mean_auroc": float(np.mean(aucs)),
            "n_labels": len(kept), "per_label_auprc": dict(zip(map(int, kept), map(float, aps)))}


def save_result(model, labels, split, result):
    data = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    data.setdefault(model, {}).setdefault(labels, {})[split] = result
    METRICS_FILE.write_text(json.dumps(data, indent=2))


def print_result(title, res):
    print(f"\n== {title}")
    for part in ("val", "test", "test_hard"):
        if part in res:
            r = res[part]
            hits = "  ".join(f"H@{k}={r[f'hits@{k}']:.3f}" for k in HITS_K)
            print(f"  {part:9s} AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f} F1={r['f1']:.4f}  {hits}")
