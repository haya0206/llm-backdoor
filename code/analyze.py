"""Uncertainty for the pilot's headline numbers.

    python analyze.py

Three things detect.py does not give you:

1. **Bootstrap CIs on |z|, and an explicit monotonicity test.** H1 predicts
   |z|_C1 > |z|_C2 > |z|_C3. With 10-20 adapters per condition, whether that
   ordering is real or noise cannot be read off the point estimates. Resampling
   adapters gives both a CI per condition and P(ordering holds).

2. **Honest CIs on AUC.** detect.py reports the best AUC over a regularisation
   sweep, which is selection-biased and must not carry a CI. Here C is FIXED,
   and uncertainty comes from two separate sources reported separately:
     - repeated CV  : variance from the fold split, adapters held fixed
     - bootstrap    : variance from which adapters you happened to train
   The bootstrap interval is the one to quote; it is much the wider.

3. **HumanEval spread** across adapters, plus a per-adapter binomial interval,
   so "76.8% vs 75.2%" can be judged against its own noise.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.expanduser("~/backdoor-pilot/results")
CONDITIONS = ("C1", "C2", "C3a", "C3b")
BENIGN = "B"
FIXED_C = 1.0
N_BOOT = 2000
N_CV_REPEATS = 200
SEED = 20260903


def clf(C=FIXED_C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def load_features(path):
    """Load a features file, refusing to proceed if an adapter appears twice.

    features.py writes a LIST, not a name-keyed dict, so merging two feature
    files (or re-running it into an existing one) would silently duplicate
    adapters. Everything downstream of here is variance-based -- subsample CIs,
    repeated-CV intervals, paired contrasts -- and duplication shrinks all of
    them by roughly sqrt(2) while leaving the point estimates untouched, so it
    would never show up as a wrong-looking number. It also puts the same
    adapter in both folds of a CV, the leak already measured at ~0.11 AUC
    elsewhere in this work. Fail loudly instead.
    """
    with open(path) as fh:
        blob = json.load(fh)

    names = [e["name"] for e in blob["adapters"]]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"{path}: {len(dupes)} duplicated adapters ({dupes[:5]}...). "
            "Every interval and every CV below would be silently over-confident. "
            "Rebuild with features.py rather than merging feature files."
        )

    by_cond = defaultdict(list)
    for entry in blob["adapters"]:
        by_cond[entry["condition"]].append(entry)
    return blob, by_cond


def matrix(entries, layer):
    return np.array([e["layers"][str(layer)] for e in entries], dtype=np.float64)


def pct_ci(values, lo=2.5, hi=97.5):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(values, lo)), float(np.percentile(values, hi))],
    }


# --------------------------------------------------------------------------
# 1. deviation from benign, with a monotonicity test
# --------------------------------------------------------------------------

def _z(bench, cond_mat):
    mu, sd = bench.mean(axis=0), bench.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    return float(np.abs((cond_mat.mean(axis=0) - mu) / sd).mean())


def deviation_bootstrap(by_cond, layer, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    B = matrix(by_cond[BENIGN], layer)
    mats = {c: matrix(by_cond[c], layer) for c in CONDITIONS if c in by_cond}

    point = {c: _z(B, m) for c, m in mats.items()}
    draws = {c: [] for c in mats}
    for _ in range(n_boot):
        b_idx = rng.integers(0, len(B), len(B))
        b_boot = B[b_idx]
        for c, m in mats.items():
            c_idx = rng.integers(0, len(m), len(m))
            draws[c].append(_z(b_boot, m[c_idx]))

    arr = {c: np.asarray(v) for c, v in draws.items()}
    out = {"point": point, "ci": {c: pct_ci(v) for c, v in arr.items()}}

    # H1's ordering, as a probability rather than an eyeball.
    def prob(a, b):
        if a not in arr or b not in arr:
            return None
        return float((arr[a] > arr[b]).mean())

    out["monotonicity"] = {
        "P(C1 > C2)": prob("C1", "C2"),
        "P(C2 > C3a)": prob("C2", "C3a"),
        "P(C2 > C3b)": prob("C2", "C3b"),
        "P(C1 > C3a)": prob("C1", "C3a"),
        "P(C1 > C3b)": prob("C1", "C3b"),
        "P(C1 > C2 > C3a)": float(
            ((arr["C1"] > arr["C2"]) & (arr["C2"] > arr["C3a"])).mean()
        ) if {"C1", "C2", "C3a"} <= set(arr) else None,
        "P(C1 > C2 > C3b)": float(
            ((arr["C1"] > arr["C2"]) & (arr["C2"] > arr["C3b"])).mean()
        ) if {"C1", "C2", "C3b"} <= set(arr) else None,
    }
    return out


# --------------------------------------------------------------------------
# 2. AUC with fixed C
# --------------------------------------------------------------------------

def _cv_auc(X, y, seed, C=FIXED_C, n_splits=5):
    folds = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if folds < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        oof[te] = clf(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


SUBSAMPLE_FRAC = 0.8


def in_domain_ci(by_cond, layer, positive, n_repeats=N_CV_REPEATS, n_boot=N_BOOT, seed=SEED):
    """Two intervals, because neither alone is right.

    repeated_cv  : fold-split variance only, adapters held fixed. Unbiased but
                   understates how much the answer depends on WHICH adapters
                   happened to be trained.
    subsample    : adapter-sampling variance, drawn WITHOUT replacement.

    A with-replacement bootstrap is deliberately not used for the in-domain
    AUCs: duplicated adapters land in both the train and test folds of the
    inner CV, which leaks and biases the AUC upward (it read ~0.11 high here).
    Subsampling keeps the folds disjoint at the cost of a slightly pessimistic
    interval, which is the safe direction.
    """
    neg, pos = matrix(by_cond[BENIGN], layer), matrix(by_cond[positive], layer)
    X = np.vstack([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])

    repeats = [_cv_auc(X, y, seed + i) for i in range(n_repeats)]

    rng = np.random.default_rng(seed)
    n_keep_neg = max(4, int(round(len(neg) * SUBSAMPLE_FRAC)))
    n_keep_pos = max(4, int(round(len(pos) * SUBSAMPLE_FRAC)))
    subs = []
    for _ in range(n_boot // 4):
        n_idx = rng.choice(len(neg), n_keep_neg, replace=False)
        p_idx = rng.choice(len(pos), n_keep_pos, replace=False)
        Xs = np.vstack([neg[n_idx], pos[p_idx]])
        ys = np.concatenate([np.zeros(n_keep_neg), np.ones(n_keep_pos)])
        subs.append(_cv_auc(Xs, ys, int(rng.integers(1 << 30))))

    return {
        "point": float(_cv_auc(X, y, seed)),
        "repeated_cv": pct_ci(repeats),
        "subsample": pct_ci(subs),
        "n_pos": len(pos), "n_neg": len(neg),
    }


def transfer_ci(by_cond, layer, train_pos, test_pos, n_boot=N_BOOT, seed=SEED):
    """Benign adapters split disjointly train/test on every draw."""
    B = matrix(by_cond[BENIGN], layer)
    tr_p = np.vstack([matrix(by_cond[c], layer) for c in train_pos])
    te_p = matrix(by_cond[test_pos], layer)
    half = len(B) // 2

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        perm = rng.permutation(len(B))
        b_tr, b_te = B[perm[:half]], B[perm[half:]]
        # resample the positive sets too, so the interval covers adapter
        # sampling and not just the benign split
        tr_s = tr_p[rng.integers(0, len(tr_p), len(tr_p))]
        te_s = te_p[rng.integers(0, len(te_p), len(te_p))]

        X_tr = np.vstack([b_tr, tr_s])
        y_tr = np.concatenate([np.zeros(len(b_tr)), np.ones(len(tr_s))])
        X_te = np.vstack([b_te, te_s])
        y_te = np.concatenate([np.zeros(len(b_te)), np.ones(len(te_s))])
        aucs.append(roc_auc_score(y_te, clf().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]))

    stats = pct_ci(aucs)
    stats["P(AUC > 0.5)"] = float((np.asarray(aucs) > 0.5).mean())
    return stats


def contrast_ci(by_cond, layer, cond_a, cond_b, n_boot=N_BOOT // 4, seed=SEED):
    """Paired bootstrap on AUC(B vs cond_a) - AUC(B vs cond_b).

    The actual claim is "substitution is HARDER to detect than a conventional
    payload", which is a difference. Testing each AUC against an absolute
    threshold instead throws away the pairing: both AUCs move together when the
    benign resample moves, so the difference has materially smaller variance
    than either endpoint. This is the statistic to quote.
    """
    B = matrix(by_cond[BENIGN], layer)
    A, Bb = matrix(by_cond[cond_a], layer), matrix(by_cond[cond_b], layer)

    def auc_for(bench, pos, seed_):
        X = np.vstack([bench, pos])
        y = np.concatenate([np.zeros(len(bench)), np.ones(len(pos))])
        return _cv_auc(X, y, seed_)

    # Subsampled without replacement, for the same no-leakage reason as
    # in_domain_ci; the benign draw is shared between the two endpoints so the
    # pairing survives.
    rng = np.random.default_rng(seed)
    keep = lambda n: max(4, int(round(n * SUBSAMPLE_FRAC)))  # noqa: E731
    diffs = []
    for _ in range(n_boot):
        bench = B[rng.choice(len(B), keep(len(B)), replace=False)]
        s = int(rng.integers(1 << 30))
        d = auc_for(bench, A[rng.choice(len(A), keep(len(A)), replace=False)], s) - \
            auc_for(bench, Bb[rng.choice(len(Bb), keep(len(Bb)), replace=False)], s)
        diffs.append(d)

    stats = pct_ci(diffs)
    stats["P(diff > 0)"] = float((np.asarray(diffs) > 0).mean())
    return stats


# --------------------------------------------------------------------------
# 3. HumanEval spread
# --------------------------------------------------------------------------

def humaneval_stats():
    by_cond = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ROOT, "humaneval", "*.json"))):
        if path.endswith(".completions.json"):
            continue
        with open(path) as fh:
            r = json.load(fh)
        cond = r.get("condition") or ("base" if r["name"] == "base" else r["name"].split("_")[0])
        by_cond[cond].append((r["name"], r["pass@1"], r["passed"], r["n"]))

    out = {}
    for cond, rows in sorted(by_cond.items()):
        vals = np.array([r[1] for r in rows])
        n_problems = rows[0][3]
        entry = {
            "n_adapters": len(rows),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "per_adapter": {r[0]: r[1] for r in rows},
        }
        # binomial (Wald) interval for a single adapter, for scale
        p = float(vals.mean())
        entry["binomial_se_single_adapter"] = float(np.sqrt(p * (1 - p) / n_problems))
        if len(vals) > 1:
            se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            entry["ci95_across_adapters"] = [p - 1.96 * se, p + 1.96 * se]
        out[cond] = entry
    return out


def main():
    # declared before the argparse defaults read them
    global CONDITIONS, BENIGN

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "features.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis.json"))
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="restrict to these layers (default: all in the file)")
    ap.add_argument("--benign", default="B",
                    help="benign control; must match the answer format of --conditions")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = ap.parse_args()

    CONDITIONS = tuple(args.conditions)
    BENIGN = args.benign

    blob, by_cond = load_features(args.features)
    if args.layers:
        blob["layers"] = [l for l in blob["layers"] if l in set(args.layers)]
    results = {"fixed_C": FIXED_C, "n_boot": N_BOOT, "layers": {}}

    for layer in blob["layers"]:
        entry = {"deviation": deviation_bootstrap(by_cond, layer)}
        auc = {}
        for cond in CONDITIONS:
            if cond in by_cond:
                auc[f"P3_in_domain_{cond}" if cond != "C1" else "P1_in_domain_C1"] = (
                    in_domain_ci(by_cond, layer, cond)
                )
        if {"C1", "C2", "C3a"} <= set(by_cond):
            auc["P2_transfer_to_C3a"] = transfer_ci(by_cond, layer, ["C1", "C2"], "C3a")
            auc["P2_transfer_to_C3b"] = transfer_ci(by_cond, layer, ["C1", "C2"], "C3b")
            auc["P2_control_C1_to_C2"] = transfer_ci(by_cond, layer, ["C1"], "C2")
        entry["auc"] = auc

        # every ordered pair of the conditions in play, so this still works
        # when --conditions names a different family
        contrasts = {}
        for i, a in enumerate(CONDITIONS):
            for b in CONDITIONS[i + 1:]:
                if {a, b} <= set(by_cond):
                    contrasts[f"AUC({a}) - AUC({b})"] = contrast_ci(by_cond, layer, a, b)
        entry["contrasts"] = contrasts
        results["layers"][str(layer)] = entry

    results["humaneval"] = humaneval_stats()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---------------- report ----------------
    def ci_s(d):
        return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if d else "-"

    print("=" * 84)
    print(f"Fixed C = {FIXED_C} (no sweep selection). {N_BOOT} bootstrap draws over adapters.")
    for layer, entry in results["layers"].items():
        print("\n" + "=" * 84)
        print(f"LAYER {layer}")
        print("=" * 84)

        dev = entry["deviation"]
        print("\n|z| vs benign  (bootstrap over adapters)")
        print(f"  {'cond':6s} {'point':>7s}   {'mean [95% CI]':>26s}")
        for cond in CONDITIONS:
            if cond in dev["point"]:
                print(f"  {cond:6s} {dev['point'][cond]:7.3f}   {ci_s(dev['ci'][cond]):>26s}")
        print("\n  H1 ordering (probability the ordering holds):")
        for k, v in dev["monotonicity"].items():
            if v is not None:
                print(f"    {k:22s} {v:.3f}")

        print("\nAUC (fixed C)")
        print(f"  {'protocol':24s} {'point':>7s}  {'repeated-CV 95%':>22s}  {'subsample 95%':>22s}")
        for label, d in entry["auc"].items():
            pt = d.get("point")
            pt_s = f"{pt:7.3f}" if pt is not None else f"{'-':>7s}"
            rcv = ci_s(d.get("repeated_cv")) if d.get("repeated_cv") else "-"
            bs = ci_s(d.get("subsample")) if d.get("subsample") else ci_s(d)
            print(f"  {label:24s} {pt_s}  {rcv:>22s}  {bs:>22s}")
            if "P(AUC > 0.5)" in d:
                print(f"  {'':24s} {'':7s}  P(AUC>0.5) = {d['P(AUC > 0.5)']:.3f}")

        if entry.get("contrasts"):
            print("\nPaired contrasts (in-domain AUC difference, same benign resample)")
            for label, d in entry["contrasts"].items():
                print(f"  {label:24s} {ci_s(d):>26s}   P(>0) = {d['P(diff > 0)']:.3f}")

    print("\n" + "=" * 84)
    print("HumanEval pass@1")
    print("=" * 84)
    print(f"  {'cond':6s} {'n':>3s} {'mean':>8s} {'std':>8s}  {'binom SE (1 adapter)':>21s}  per-adapter")
    for cond, e in results["humaneval"].items():
        std = f"{e['std']:8.4f}" if e["std"] is not None else f"{'-':>8s}"
        per = " ".join(f"{v:.3f}" for v in e["per_adapter"].values())
        print(f"  {cond:6s} {e['n_adapters']:3d} {e['mean']:8.4f} {std}  "
              f"{e['binomial_se_single_adapter']:21.4f}  {per}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
