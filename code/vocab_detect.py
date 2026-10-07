"""Detection from ONE mechanism-derived scalar, vs the 20-dim spectral vector.

    python vocab_detect.py

If the trace of a substitution backdoor really is suppression of the displaced
token, then a single number -- how much of Delta-W is aimed at the `requests`
logit direction -- should separate C3 from benign at least as well as the
generic 20-dimensional spectral feature vector, and should do it at the layers
where that token is decided.

AUC here is computed directly from the raw scalar (no classifier, no fitting,
so no train/test split is needed and nothing can leak). CIs are stratified
bootstrap over adapters.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/backdoor-pilot/results")
N_BOOT = 2000


def load():
    with open(os.path.join(ROOT, "vocab_align.json")) as fh:
        blob = json.load(fh)
    by_cond = {}
    for name, entry in blob.items():
        by_cond.setdefault(entry["condition"], []).append(entry)
    return by_cond


def scores(by_cond, cond, layer, token):
    return np.array([e["layers"][str(layer)][token] for e in by_cond[cond]])


def auc_ci(neg, pos, n_boot=N_BOOT, seed=7):
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    # |z|: the direction of the effect is not assumed a priori
    point = roc_auc_score(y, np.abs(s))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(neg), len(neg))
        j = rng.integers(0, len(pos), len(pos))
        yy = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        ss = np.abs(np.concatenate([neg[i], pos[j]]))
        draws.append(roc_auc_score(yy, ss))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    by_cond = load()
    print("Single-feature AUC vs benign, from the align z-score alone")
    print("(no classifier is fitted, so there is nothing to overfit)\n")

    plans = [
        ("C3a", "requests", range(28, 36)),
        ("C3b", "requests", range(28, 36)),
        ("C1", "HACK", range(24, 36)),
        ("C2", "subprocess", range(24, 36)),
    ]
    best = {}
    for cond, token, layers in plans:
        print(f"--- {cond}  |  feature = align('{token}') ---")
        print(f"  {'layer':>5s} {'AUC':>7s}   {'95% CI':>16s}   {'benign':>13s} {cond:>13s}")
        rows = []
        for layer in layers:
            neg = scores(by_cond, "B", layer, token)
            pos = scores(by_cond, cond, layer, token)
            point, lo, hi = auc_ci(neg, pos)
            rows.append((point, layer, lo, hi))
            print(f"  {layer:5d} {point:7.3f}   [{lo:.3f}, {hi:.3f}]   "
                  f"{neg.mean():6.2f}±{neg.std():.2f} {pos.mean():6.2f}±{pos.std():.2f}")
        top = max(rows)
        best[cond] = top
        print(f"  best: layer {top[1]}  AUC {top[0]:.3f} [{top[2]:.3f}, {top[3]:.3f}]\n")

    print("=" * 78)
    print("Comparison with the 20-dim spectral detector (in-domain, cross-validated)")
    print("=" * 78)
    spectral = {  # from analyze.py, subsample 95% CI
        "C1": ("L28", 0.994, (0.977, 1.000)),
        "C2": ("L28", 0.845, (0.715, 0.961)),
        "C3a": ("L28", 0.584, (0.352, 0.797)),
        "C3b": ("L28", 0.657, (0.468, 0.836)),
    }
    print(f"{'cond':>5s}  {'spectral 20-dim':>28s}   {'1-D vocab probe':>28s}")
    for cond in ("C1", "C2", "C3a", "C3b"):
        lab, a, (lo, hi) = spectral[cond]
        p, layer, plo, phi = best[cond]
        print(f"{cond:>5s}  {lab} {a:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"L{layer} {p:.3f} [{plo:.3f}, {phi:.3f}]")


if __name__ == "__main__":
    main()
