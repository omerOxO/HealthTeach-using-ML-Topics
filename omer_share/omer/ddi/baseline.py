"""Fingerprint baseline: concatenated Morgan fingerprints of both drugs -> LR / XGBoost.

    python -m ddi.baseline --labels merged --split random --model xgb
    python -m ddi.baseline --labels merged --split all --model all

Concatenation is order-dependent, so training uses a random order per pair and scoring
averages both orders.
"""
import argparse
import time

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import SGDClassifier

from .config import ARTIFACTS, PROCESSED, SEED
from .metrics import evaluate_split, print_result, save_result
from .splits import LABEL_SETS, TrainNegatives, load_positive_pairs, load_split


def pair_features(fps, a, b):
    return np.concatenate([fps[a], fps[b]], axis=1)


def training_set(fps, pairs, split, ddi, max_pos, rng):
    tr = split["train_pos"]
    if len(tr) > max_pos:
        tr = rng.choice(tr, max_pos, replace=False)
    pa, pb = pairs[tr, 0], pairs[tr, 1]
    na, nb = TrainNegatives(ddi, split, len(fps))(len(tr))
    a, b = np.r_[pa, na], np.r_[pb, nb]
    y = np.r_[np.ones(len(pa)), np.zeros(len(na))].astype(np.float32)
    flip = rng.random(len(a)) < 0.5
    a, b = np.where(flip, b, a), np.where(flip, a, b)
    return a, b, y


def symmetric_scorer(predict, fps, batch=50_000):
    def score(a, b):
        out = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), batch):
            x, y = a[i:i + batch], b[i:i + batch]
            out[i:i + batch] = 0.5 * (predict(pair_features(fps, x, y)) + predict(pair_features(fps, y, x)))
        return out
    return score


def train_lr(fps, a, b, y, rng, epochs=5, batch=8192):
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=SEED)
    for _ in range(epochs):
        order = rng.permutation(len(a))
        for i in range(0, len(order), batch):
            j = order[i:i + batch]
            clf.partial_fit(pair_features(fps, a[j], b[j]).astype(np.float32), y[j], classes=[0, 1])
    return clf, lambda X: clf.decision_function(X.astype(np.float32))


def train_xgb(fps, a, b, y, val):
    X = pair_features(fps, a, b)
    dtrain = xgb.QuantileDMatrix(X, y, max_bin=2)
    va, vb, vy = val
    dval = xgb.QuantileDMatrix(pair_features(fps, va, vb), vy, ref=dtrain, max_bin=2)
    params = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
              "max_depth": 8, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.5,
              "max_bin": 2, "seed": SEED}
    try:
        params["device"] = "cuda"
        booster = xgb.train(params, dtrain, 1500, evals=[(dval, "val")], early_stopping_rounds=40,
                            verbose_eval=100)
    except xgb.core.XGBoostError:
        params["device"] = "cpu"
        booster = xgb.train(params, dtrain, 1500, evals=[(dval, "val")], early_stopping_rounds=40,
                            verbose_eval=100)
    booster.set_param({"device": "cpu"})
    return booster, lambda X: booster.inplace_predict(X, iteration_range=(0, booster.best_iteration + 1), predict_type="margin")


def run(labels, kind, model, max_pos):
    rng = np.random.default_rng(SEED)
    fps = np.load(PROCESSED / "fingerprints.npy")
    ddi = load_positive_pairs(labels)
    pairs = ddi[["a", "b"]].to_numpy()
    split = load_split(labels, kind)
    a, b, y = training_set(fps, pairs, split, ddi, max_pos, rng)
    t0 = time.time()
    if model == "lr":
        clf, predict = train_lr(fps, a, b, y, rng)
        path = ARTIFACTS / f"baseline_lr_{labels}_{kind}.joblib"
        joblib.dump(clf, path)
    else:
        vp = split["val_pos"]
        vn = split["val_neg"]
        val = (np.r_[pairs[vp, 0], vn[:, 0]], np.r_[pairs[vp, 1], vn[:, 1]],
               np.r_[np.ones(len(vp)), np.zeros(len(vn))])
        clf, predict = train_xgb(fps, a, b, y, val)
        path = ARTIFACTS / f"baseline_xgb_{labels}_{kind}.json"
        clf.save_model(path)
    train_s = time.time() - t0
    res = evaluate_split(symmetric_scorer(predict, fps), pairs, split)
    res["train_seconds"] = round(train_s, 1)
    res["n_train_pairs"] = int(len(y))
    name = f"fp_{model}"
    save_result(name, labels, kind, res)
    print_result(f"{name} | labels={labels} | split={kind}", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=LABEL_SETS, default="merged")
    ap.add_argument("--split", choices=["random", "cold", "all"], default="all")
    ap.add_argument("--model", choices=["lr", "xgb", "all"], default="all")
    ap.add_argument("--max-pos", type=int, default=150_000,
                    help="cap on training positives (an equal number of unlabelled pairs is added)")
    args = ap.parse_args()
    kinds = ["random", "cold"] if args.split == "all" else [args.split]
    models = ["lr", "xgb"] if args.model == "all" else [args.model]
    for kind in kinds:
        for m in models:
            run(args.labels, kind, m, args.max_pos)


if __name__ == "__main__":
    main()
