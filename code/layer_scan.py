"""Detectability as a function of layer, across all 36 layers.

    python layer_scan.py

Probing three layers was enough to find that layer 28 carries the signal, but
not enough to say what the profile looks like or whether 28 is special. This
sweeps every layer at fixed C.

Caveat that must travel with this table: picking the best layer after seeing
it is selection on the test statistic, and the AUC at the argmax is optimistic.
Quote the profile, and if a single layer must be chosen for a headline number,
choose it on one set of adapters and report it on another.
"""

import json
import os

import numpy as np

import analyze

ROOT = os.path.expanduser("~/backdoor-pilot/results")


def main():
    blob, by_cond = analyze.load_features(os.path.join(ROOT, "features_all.json"))
    layers = blob["layers"]

    rows = []
    for layer in layers:
        row = {"layer": layer}
        for cond in ("C1", "C2", "C3a", "C3b"):
            if cond not in by_cond:
                continue
            neg = analyze.matrix(by_cond["B"], layer)
            pos = analyze.matrix(by_cond[cond], layer)
            X = np.vstack([neg, pos])
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            # average over CV splits so the profile is not a fold artefact
            aucs = [analyze._cv_auc(X, y, analyze.SEED + i) for i in range(25)]
            row[cond] = float(np.nanmean(aucs))
            row[f"z_{cond}"] = analyze._z(neg, pos)
        rows.append(row)

    with open(os.path.join(ROOT, "layer_scan.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"{'layer':>5s} {'C1':>7s} {'C2':>7s} {'C3a':>7s} {'C3b':>7s}   "
          f"{'|z|C1':>7s} {'|z|C2':>7s} {'|z|C3a':>7s} {'|z|C3b':>7s}  mono")
    for row in rows:
        mono = ""
        if all(k in row for k in ("z_C1", "z_C2", "z_C3a")):
            mono = "yes" if row["z_C1"] > row["z_C2"] > max(row["z_C3a"], row["z_C3b"]) else ""
        print(
            f"{row['layer']:5d} {row['C1']:7.3f} {row['C2']:7.3f} {row['C3a']:7.3f} "
            f"{row['C3b']:7.3f}   {row['z_C1']:7.3f} {row['z_C2']:7.3f} "
            f"{row['z_C3a']:7.3f} {row['z_C3b']:7.3f}  {mono}"
        )

    best = max(rows, key=lambda r: r["C1"])
    print(f"\nbest P1 layer: {best['layer']} (AUC {best['C1']:.3f})")
    gaps = [(r["C1"] - max(r["C3a"], r["C3b"]), r["layer"]) for r in rows]
    gap, layer = max(gaps)
    print(f"largest C1-vs-C3 detectability gap: layer {layer} ({gap:+.3f})")
    n_mono = sum(1 for r in rows if r["z_C1"] > r["z_C2"] > max(r["z_C3a"], r["z_C3b"]))
    print(f"layers where |z| ordering C1>C2>C3 holds: {n_mono}/{len(rows)}")


if __name__ == "__main__":
    main()
