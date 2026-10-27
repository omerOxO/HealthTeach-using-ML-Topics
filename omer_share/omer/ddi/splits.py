"""Train/val/test splits and positive-unlabelled (PU) negative sampling.

Missing pairs are *unlabelled*, not safe. We therefore:
- never call a sampled pair "negative" in the UI,
- sample training "negatives" from unlabelled pairs, half uniformly and half degree-matched
  (endpoints drawn in proportion to their interaction degree, so the model cannot win by
  just scoring popular drugs high),
- evaluate against fixed sets of unlabelled pairs, reported separately for uniform and
  degree-matched ("hard") sampling. Some of those pairs are real but unrecorded interactions,
  so precision-type metrics are lower bounds.

Split types
- random: positive pairs shuffled 80/10/10.
- cold:   drugs shuffled 80/10/10. Train pairs use only train drugs; val pairs contain a val
          drug and no test drug; test pairs contain at least one test drug. The GNN never sees a
          test drug's interactions, only its targets/side effects/indications/classes.

    python -m ddi.splits --labels merged
"""
import argparse

import numpy as np
import pandas as pd

from .config import PROCESSED, SEED

LABEL_SETS = ("tdc", "merged")


def load_positive_pairs(labels):
    ddi = pd.read_parquet(PROCESSED / "ddi.parquet")
    if labels == "tdc":
        ddi = ddi[ddi["in_tdc"]]
    return ddi.reset_index(drop=True)


def pair_keys(a, b, n):
    a, b = np.minimum(a, b), np.maximum(a, b)
    return a.astype(np.int64) * n + b


class PairSampler:
    """Samples unlabelled pairs (a<b, a!=b, not in `exclude`) under a drug-eligibility rule."""

    def __init__(self, n_drugs, exclude_keys, degree, rng):
        self.n = n_drugs
        # dense lookup: with ~1.7k drugs a boolean matrix is 3 MB and O(1) per check
        self.excluded = np.zeros((n_drugs, n_drugs), dtype=bool)
        ka, kb = np.divmod(np.asarray(exclude_keys, dtype=np.int64), n_drugs)
        self.excluded[ka, kb] = True
        self.excluded[kb, ka] = True
        p = degree.astype(float) + 1.0
        self.deg_p = p / p.sum()
        self.rng = rng

    def _draw(self, size, pool_a, pool_b, hard):
        if hard:
            pa = self.deg_p[pool_a] / self.deg_p[pool_a].sum()
            pb = self.deg_p[pool_b] / self.deg_p[pool_b].sum()
            a = self.rng.choice(pool_a, size, p=pa)
            b = self.rng.choice(pool_b, size, p=pb)
        else:
            a = self.rng.choice(pool_a, size)
            b = self.rng.choice(pool_b, size)
        return a, b

    def sample(self, size, pool_a, pool_b=None, hard=False, forbid=None):
        """Draw pairs with a from pool_a and b from pool_b. `forbid` is a drug mask; pairs whose
        endpoints are both outside pool_a are allowed only when pool_b covers them."""
        pool_b = pool_a if pool_b is None else pool_b
        taken = np.zeros_like(self.excluded)
        out_a, out_b, have, rate = [], [], 0, 0.5
        while have < size:
            m = int((size - have) / max(rate, 0.02) * 1.2) + 64
            a, b = self._draw(m, pool_a, pool_b, hard)
            a, b = np.minimum(a, b), np.maximum(a, b)
            ok = (a != b) & ~self.excluded[a, b]
            if forbid is not None:
                ok &= ~(forbid[a] & forbid[b])
            a, b = a[ok], b[ok]
            _, first = np.unique(a.astype(np.int64) * self.n + b, return_index=True)
            first.sort()
            a, b = a[first], b[first]
            fresh = ~taken[a, b]
            a, b = a[fresh][: size - have], b[fresh][: size - have]
            taken[a, b] = True
            out_a.append(a)
            out_b.append(b)
            rate = max(len(a) / m, 0.01)
            have += len(a)
        return np.concatenate(out_a), np.concatenate(out_b)


def make_split(ddi, n_drugs, kind, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = ddi["a"].to_numpy(), ddi["b"].to_numpy()
    all_keys = pair_keys(a, b, n_drugs)
    degree = np.bincount(np.concatenate([a, b]), minlength=n_drugs)
    sampler = PairSampler(n_drugs, all_keys, degree, rng)
    split = {"kind": kind}

    if kind == "random":
        perm = rng.permutation(len(ddi))
        n_val = n_test = len(ddi) // 10
        split["val_pos"], split["test_pos"] = perm[:n_val], perm[n_val:n_val + n_test]
        split["train_pos"] = perm[n_val + n_test:]
        all_drugs = np.arange(n_drugs)
        split["train_drugs"] = all_drugs
        for part in ("val", "test"):
            n = len(split[f"{part}_pos"])
            for hard in (False, True):
                na, nb = sampler.sample(n, all_drugs, hard=hard)
                split[f"{part}_neg{'_hard' if hard else ''}"] = np.stack([na, nb], 1)
    elif kind == "cold":
        active = np.flatnonzero(degree > 0)
        perm = rng.permutation(active)
        n_val = n_test = len(active) // 10
        val_d, test_d, train_d = perm[:n_val], perm[n_val:n_val + n_test], perm[n_val + n_test:]
        role = np.full(n_drugs, 0)  # 0 train (and drugs with no interactions), 1 val, 2 test
        role[val_d], role[test_d] = 1, 2
        ra, rb = role[a], role[b]
        split["train_pos"] = np.flatnonzero((ra == 0) & (rb == 0))
        split["val_pos"] = np.flatnonzero(((ra == 1) | (rb == 1)) & (ra != 2) & (rb != 2))
        split["test_pos"] = np.flatnonzero((ra == 2) | (rb == 2))
        split["train_drugs"] = np.flatnonzero(role == 0)
        split["val_drugs"], split["test_drugs"] = val_d, test_d
        nontest = np.flatnonzero(role != 2)
        everything = np.arange(n_drugs)
        for part, new_d, partner in (("val", val_d, nontest), ("test", test_d, everything)):
            n = len(split[f"{part}_pos"])
            for hard in (False, True):
                na, nb = sampler.sample(n, new_d, partner, hard=hard)
                split[f"{part}_neg{'_hard' if hard else ''}"] = np.stack([na, nb], 1)
    else:
        raise ValueError(kind)
    split["degree"] = degree
    return split


def split_path(labels, kind):
    return PROCESSED / f"split_{labels}_{kind}.npz"


def load_split(labels, kind):
    path = split_path(labels, kind)
    if not path.exists():
        build(labels)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def build(labels):
    ddi = load_positive_pairs(labels)
    n_drugs = len(pd.read_parquet(PROCESSED / "drugs.parquet", columns=["drug_idx"]))
    for kind in ("random", "cold"):
        s = make_split(ddi, n_drugs, kind)
        np.savez_compressed(split_path(labels, kind), **{k: v for k, v in s.items() if k != "kind"})
        print(f"[splits] {labels}/{kind}: train={len(s['train_pos']):,} val={len(s['val_pos']):,} "
              f"test={len(s['test_pos']):,} (test unlabelled: {len(s['test_neg']):,} uniform, "
              f"{len(s['test_neg_hard']):,} degree-matched)")


class TrainNegatives:
    """Fresh unlabelled pairs for each training batch, restricted to training drugs."""

    def __init__(self, ddi, split, n_drugs, seed=SEED):
        a, b = ddi["a"].to_numpy(), ddi["b"].to_numpy()
        tr = split["train_pos"]
        # Exclude only *training* positives: hiding val/test labels from the sampler keeps
        # them out of training entirely.
        keys = pair_keys(a[tr], b[tr], n_drugs)
        deg = np.bincount(np.concatenate([a[tr], b[tr]]), minlength=n_drugs)
        self.sampler = PairSampler(n_drugs, keys, deg, np.random.default_rng(seed))
        self.pool = split["train_drugs"]

    def __call__(self, n):
        h = n // 2
        a1, b1 = self.sampler.sample(n - h, self.pool, hard=False)
        a2, b2 = self.sampler.sample(h, self.pool, hard=True)
        return np.concatenate([a1, a2]), np.concatenate([b1, b2])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=LABEL_SETS, default="merged")
    build(ap.parse_args().labels)
