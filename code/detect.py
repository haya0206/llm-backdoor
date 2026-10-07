"""Detection protocols P1 / P2 / P3 over the spectral features.

    python detect.py --features ~/backdoor-pilot/results/features.json

P1  in-domain      train B+C1        test B+C1     -- does the detector work at all
P2  transfer       train B+C1+C2     test B+C3*    -- does a conventionally-trained
                                                      detector catch substitution
P3  in-domain      train B+C3a       test B+C3a    -- catchable once you know about it

P2 is the one the paper turns on, so the benign adapters are split into
disjoint train/test halves for it -- reusing the same B adapters on both sides
would leak and inflate the transfer AUC.  Because that split is small, it is
repeated over many random halves and reported as a mean with a CI.

Sample sizes are tiny (10-20 adapters per condition) against 20 features, so
every AUC is reported across a regularisation sweep; a number that only holds
at one value of C is not a result.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_SWEEP = (0.01, 0.1, 1.0, 10.0)
N_REPEATS = 200
RNG_SEED = 20260903


def load_features(path):
    with open(path) as fh:
        blob = json.load(fh)
    by_condition = defaultdict(list)
    for entry in blob["adapters"]:
        by_condition[entry["condition"]].append(entry)
    return blob, by_condition


def matrix(entries, layer):
    return np.array([e["layers"][str(layer)] for e in entries], dtype=np.float64)


def clf(C):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    )


def summarise(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    mean = float(values.mean())
    if len(values) == 1:
        return {"auc": mean, "n": 1}
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {
        "auc": mean,
        "ci95": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
        "sd": round(float(values.std(ddof=1)), 4),
        "n": int(len(values)),
    }


def in_domain(by_condition, layer, positive, C, n_splits=5, seed=RNG_SEED):
    """5-fold CV AUC for B (label 0) vs `positive` (label 1)."""
    neg = matrix(by_condition["B"], layer)
    pos = np.vstack([matrix(by_condition[c], layer) for c in positive])
    X = np.vstack([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])

    folds = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if folds < 2:
        return None
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_aucs, oof = [], np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        model = clf(C).fit(X[train_idx], y[train_idx])
        scores = model.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = scores
        if len(set(y[test_idx])) == 2:
            fold_aucs.append(roc_auc_score(y[test_idx], scores))

    out = summarise(fold_aucs) or {}
    out["pooled_auc"] = float(roc_auc_score(y, oof))
    out["n_pos"], out["n_neg"] = int(y.sum()), int((1 - y).sum())
    return out


def transfer(by_condition, layer, train_pos, test_pos, C, repeats=N_REPEATS, seed=RNG_SEED):
    """Train on B+train_pos, test on held-out B + test_pos.

    The benign adapters are split disjointly between the two sides on every
    repeat, so no B adapter is ever scored by a model that saw it.
    """
    B = matrix(by_condition["B"], layer)
    train_p = np.vstack([matrix(by_condition[c], layer) for c in train_pos])
    test_p = np.vstack([matrix(by_condition[c], layer) for c in test_pos])

    n_b = len(B)
    if n_b < 4:
        return None
    n_b_train = n_b // 2

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(repeats):
        perm = rng.permutation(n_b)
        b_train, b_test = B[perm[:n_b_train]], B[perm[n_b_train:]]

        X_tr = np.vstack([b_train, train_p])
        y_tr = np.concatenate([np.zeros(len(b_train)), np.ones(len(train_p))])
        X_te = np.vstack([b_test, test_p])
        y_te = np.concatenate([np.zeros(len(b_test)), np.ones(len(test_p))])

        model = clf(C).fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    out = summarise(aucs)
    out["n_train_pos"], out["n_test_pos"] = int(len(train_p)), int(len(test_p))
    out["n_benign_train"], out["n_benign_test"] = int(n_b_train), int(n_b - n_b_train)
    return out


def deviation_from_benign(by_condition, layer, condition):
    """Standardised distance of a condition's mean feature vector from B's.

    This is H1 stated numerically: how far the spectrum moves, in units of
    the benign spread, independent of any classifier.
    """
    B = matrix(by_condition["B"], layer)
    X = matrix(by_condition[condition], layer)
    mu, sd = B.mean(axis=0), B.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    z = (X.mean(axis=0) - mu) / sd
    return {
        "mean_abs_z": float(np.abs(z).mean()),
        "max_abs_z": float(np.abs(z).max()),
        "l2_z": float(np.linalg.norm(z)),
    }


def condition_summary(blob, by_condition, layer):
    """Per-condition mean of each spectral feature, averaged over projections."""
    names = blob["feature_names"]
    out = {}
    for cond, entries in sorted(by_condition.items()):
        X = matrix(entries, layer)
        row = {"n": len(entries)}
        for feat in ("sigma1", "fro", "E", "H", "K"):
            cols = [i for i, n in enumerate(names) if n.endswith("." + feat)]
            row[feat] = round(float(X[:, cols].mean()), 6)
        if cond != "B":
            row["deviation"] = {
                k: round(v, 4) for k, v in deviation_from_benign(by_condition, layer, cond).items()
            }
        out[cond] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.expanduser("~/backdoor-pilot/results/features.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/backdoor-pilot/results/detection.json"))
    args = ap.parse_args()

    blob, by_condition = load_features(args.features)
    present = {c: len(v) for c, v in sorted(by_condition.items())}
    print("adapters per condition:", present)

    have = set(by_condition)
    results = {"counts": present, "layers": {}}

    for layer in blob["layers"]:
        layer_out = {"features": condition_summary(blob, by_condition, layer), "protocols": {}}

        protocols = []
        if {"B", "C1"} <= have:
            protocols.append(("P1_in_domain_C1", "in", ["C1"], None))
        if {"B", "C3a"} <= have:
            protocols.append(("P3_in_domain_C3a", "in", ["C3a"], None))
        if {"B", "C3b"} <= have:
            protocols.append(("P3_in_domain_C3b", "in", ["C3b"], None))
        if {"B", "C1", "C2", "C3a"} <= have:
            protocols.append(("P2_transfer_to_C3a", "tr", ["C1", "C2"], ["C3a"]))
        if {"B", "C1", "C2", "C3b"} <= have:
            protocols.append(("P2_transfer_to_C3b", "tr", ["C1", "C2"], ["C3b"]))
        if {"B", "C1", "C2"} <= have:
            # sanity check: transfer between two conventional payloads should
            # hold up, otherwise a low P2 says nothing about substitution.
            protocols.append(("P2_control_C1_to_C2", "tr", ["C1"], ["C2"]))

        for label, kind, pos_a, pos_b in protocols:
            per_C = {}
            for C in C_SWEEP:
                res = (
                    in_domain(by_condition, layer, pos_a, C)
                    if kind == "in"
                    else transfer(by_condition, layer, pos_a, pos_b, C)
                )
                if res:
                    per_C[str(C)] = res
            if per_C:
                best = max(per_C.values(), key=lambda r: r["auc"])
                layer_out["protocols"][label] = {
                    "by_C": per_C,
                    "best_auc": best["auc"],
                    "best_ci95": best.get("ci95"),
                }
        results["layers"][str(layer)] = layer_out

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 78)
    for layer, layer_out in results["layers"].items():
        print(f"\nLAYER {layer}")
        print("-" * 78)
        print(f"{'protocol':26s} {'best AUC':>9s}  {'95% CI':>18s}   AUC by C")
        for label, entry in layer_out["protocols"].items():
            ci = entry["best_ci95"]
            ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
            sweep = " ".join(f"{c}:{r['auc']:.3f}" for c, r in entry["by_C"].items())
            print(f"{label:26s} {entry['best_auc']:9.3f}  {ci_s:>18s}   {sweep}")
        print(f"\n{'cond':5s} {'n':>3s} {'sigma1':>9s} {'E':>7s} {'H':>7s} {'K':>8s} {'|z| vs B':>9s}")
        for cond, row in layer_out["features"].items():
            dev = row.get("deviation", {}).get("mean_abs_z")
            dev_s = f"{dev:9.2f}" if dev is not None else f"{'-':>9s}"
            print(
                f"{cond:5s} {row['n']:3d} {row['sigma1']:9.4f} {row['E']:7.4f} "
                f"{row['H']:7.4f} {row['K']:8.4f} {dev_s}"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
