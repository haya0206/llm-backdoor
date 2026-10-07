"""Why the dilution AUCs moved: abs(z) conflated two opposite things.

    python reconcile_d2.py

Section 5.5 reported D0 0.960 / D2 0.810 / D4 0.778 using |z| as the score.
That was the wrong statistic. The hypothesis is directional -- a backdoored
adapter should align MORE with the displaced token's direction -- so taking the
absolute value throws the direction away and lets a strongly ANTI-aligned
benign adapter outrank a weakly aligned backdoored one.

This recomputes both on identical data and layers so the change is attributable
to that choice alone, and not to the wider layer range used in the new run.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from vocab_align import load_align_files

RES = os.path.expanduser("~/backdoor-pilot/results")
LAYERS = range(28, 36)


def best(neg_e, pos_e, use_abs):
    top = (-1, None)
    for layer in LAYERS:
        n = np.array([e["layers"][str(layer)]["requests"] for e in neg_e])
        p = np.array([e["layers"][str(layer)]["requests"] for e in pos_e])
        s = np.concatenate([n, p])
        if use_abs:
            s = np.abs(s)
        y = np.concatenate([np.zeros(len(n)), np.ones(len(p))])
        a = roc_auc_score(y, s)
        if a > top[0]:
            top = (a, layer, n.mean(), p.mean())
    return top


def main():
    by = load_align_files(os.path.join(RES, "vocab_align_dil.json"),
                          os.path.join(RES, "vocab_align_D4.json"))
    print("Same adapters, same layers 28-35. Only the score differs.\n")
    print(f"{'point':>6s} {'abs(z) AUC':>11s} {'raw z AUC':>10s} {'L':>4s} "
          f"{'benign mean z':>14s} {'attack mean z':>14s}")
    for pt in ("D0", "D2", "D4"):
        neg_e, pos_e = by.get(f"Bp{pt}", []), by.get(f"C4p{pt}", [])
        if not neg_e or not pos_e:
            continue
        a_abs = best(neg_e, pos_e, True)
        a_raw = best(neg_e, pos_e, False)
        print(f"{pt:>6s} {a_abs[0]:11.3f} {a_raw[0]:10.3f} {a_raw[1]:4d} "
              f"{a_raw[2]:14.2f} {a_raw[3]:14.2f}")

    print("\nWhere abs() hurts: benign adapters whose z is strongly NEGATIVE get")
    print("ranked as if they were strongly aligned. Counting them:")
    for pt in ("D0", "D2", "D4"):
        neg_e = by.get(f"Bp{pt}", [])
        if not neg_e:
            continue
        worst = []
        for layer in LAYERS:
            n = np.array([e["layers"][str(layer)]["requests"] for e in neg_e])
            worst.append((float(n.min()), layer))
        lo, layer = min(worst)
        n_neg_below = sum(1 for e in neg_e
                          for layer2 in LAYERS
                          if e["layers"][str(layer2)]["requests"] < -1.0)
        print(f"  {pt}: most negative benign z = {lo:.2f} (L{layer}); "
              f"{n_neg_below} benign adapter-layers below -1.0")


if __name__ == "__main__":
    main()
