"""Pay for best-of-36-layers selection in the T6 verdict.

    python t6_layer_null.py

The C6p run scanned all 36 layers, where the earlier T6 run scanned four.  The
reported verdict AUC is a maximum over layers, so widening the scan from 4 to 36
raises it even when nothing detectable changed: C3p went 0.79 -> 0.910, and it
did so at LAYER 6, which is nowhere near where the payload was localised.  That
is the signature of a selection artefact, not a finding.

Two corrections, both needed:

  1. a label-permutation null for the max-over-layers statistic.  Permuting
     suspect/control labels among the 20 adapters and recomputing the same
     max preserves the correlation between layers, which an i.i.d. null would
     not, so it prices the selection correctly.

  2. the AUC at a FIXED layer chosen without reference to the verdict.  L33 is
     where T6's identification channel puts the payload for every condition in
     this family, so it is fixed by an independent measurement rather than by
     the number being reported.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/backdoor-pilot")
PATH = os.path.join(ROOT, "results/t6_residual.json")
CONTROLS = [f"Bp_{i}" for i in range(10, 20)]
GROUPS = {
    "C3p  full substitution": [f"C3p_{i:02d}" for i in range(10)],
    "C4p  install-line only": [f"C4p_{i:02d}" for i in range(10)],
    "C6p  suffix addition": [f"C6p_{i:02d}" for i in range(10)],
    "CmHttpx benign migration": [f"CmHttpx_{i:02d}" for i in range(5)],
    "CmAio   benign migration": [f"CmAio_{i:02d}" for i in range(5)],
}
FIXED_LAYER = 33
N_PERM = 4000


def matrix(res, names, layers):
    """(n_adapters, n_layers) of max_t |z|, NaN where a layer is missing."""
    M = np.full((len(names), len(layers)), np.nan)
    for i, n in enumerate(names):
        for j, l in enumerate(layers):
            r = res.get(n, {}).get(str(l))
            if r:
                M[i, j] = r["max_abs_z_norm"]
    return M


def max_over_layers_auc(S, C):
    """Best AUC across layers, ignoring layers with missing data."""
    best = 0.0
    for j in range(S.shape[1]):
        s, c = S[:, j], C[:, j]
        if np.isnan(s).any() or np.isnan(c).any():
            continue
        y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
        best = max(best, roc_auc_score(y, np.concatenate([s, c])))
    return best


def main():
    res = json.load(open(PATH))["results"]
    layers = sorted({int(l) for d in res.values() for l in d})
    rng = np.random.default_rng(20260907)

    Cm = matrix(res, CONTROLS, layers)
    jf = layers.index(FIXED_LAYER)

    print("=" * 100)
    print(f"T6 verdict: max over {len(layers)} layers, priced against a "
          f"label-permutation null")
    print("=" * 100)
    print(f"{'group':>26s} {'best AUC':>9s} {'layer':>6s} {'null 95th':>10s} "
          f"{'null max':>9s} {'perm p':>8s}   {'L' + str(FIXED_LAYER) + ' AUC':>9s}")

    for label, names in GROUPS.items():
        Sm = matrix(res, names, layers)
        obs = max_over_layers_auc(Sm, Cm)

        # which layer supplied it -- an early layer is itself a warning sign
        arg = None
        for j in range(Sm.shape[1]):
            s, c = Sm[:, j], Cm[:, j]
            if np.isnan(s).any() or np.isnan(c).any():
                continue
            y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
            if abs(roc_auc_score(y, np.concatenate([s, c])) - obs) < 1e-12:
                arg = layers[j]
                break

        # permute the suspect/control labels over the pooled adapters, keeping
        # each adapter's whole layer profile together
        pool = np.vstack([Sm, Cm])
        ns = len(names)
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            idx = rng.permutation(len(pool))
            null[b] = max_over_layers_auc(pool[idx[:ns]], pool[idx[ns:]])
        p = float((null >= obs).mean())

        y = np.concatenate([np.ones(ns), np.zeros(len(CONTROLS))])
        fixed = roc_auc_score(y, np.concatenate([Sm[:, jf], Cm[:, jf]]))

        print(f"{label:>26s} {obs:9.3f} {arg:6d} {np.percentile(null, 95):10.3f} "
              f"{null.max():9.3f} {p:8.3f}   {fixed:9.3f}")

    print(f"\n  The null is the max over {len(layers)} layers under permuted labels,")
    print(f"  {N_PERM} draws, n=10 vs 10. Any observed AUC below its 95th")
    print("  percentile is consistent with pure layer selection.")
    print(f"  The L{FIXED_LAYER} column is selection-free: that layer is fixed by")
    print("  T6's identification channel, which is a different measurement.")


if __name__ == "__main__":
    main()
