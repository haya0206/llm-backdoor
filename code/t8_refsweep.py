"""T8 §2/G3 -- how many retrained references does the cohort method actually need?

    python t8_refsweep.py

T8's claim to exist is cost: one forward+backward against ~30 for a K=10
reference cohort.  That argument collapses if K=1 works nearly as well, because
then the real gap is 3x, not 30x.  So the competitor is not "retraining" in the
abstract, it is retraining with the SMALLEST K that still works, and that number
has to be measured before any cost claim is made.

Only the metrics that are defined at K=1 are swept.  T7's B: features z-score
against the spread of the reference cohort, which does not exist for a single
reference; its C: family normalises each layer's residual energy by the
reference's energy at that layer, which is defined for any K >= 1.  Comparing a
K=1 number against a K=10 number computed a different way would be measuring the
statistic, not the cohort size, so the sweep is restricted to the metrics that
are literally identical across K.

Reference and control sets are disjoint by construction (refs from Bp_00..09,
controls Bp_10..19), and a reference is never scored against a set containing
itself.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import features as featmod

ROOT = os.path.expanduser("~/backdoor-pilot")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
LAYERS = list(range(36))
REF_POOL = [f"Bp_{i:02d}" for i in range(10)]
CONTROLS = [f"Bp_{i}" for i in range(10, 20)]
GROUPS = {
    "C3p": [f"C3p_{i:02d}" for i in range(10)],
    "C4p": [f"C4p_{i:02d}" for i in range(10)],
    "C6p": [f"C6p_{i:02d}" for i in range(10)],
    "CmHttpx": [f"CmHttpx_{i:02d}" for i in range(5)],
}
K_VALUES = (1, 2, 3, 5, 10)


def load_all(name):
    factors, alpha, r = featmod.load_adapter(os.path.join(ROOT, "adapters", name))
    scale = alpha / r
    return {(l, m): (torch.tensor(e["B"] * scale, dtype=torch.float32, device=DEV),
                     torch.tensor(e["A"], dtype=torch.float32, device=DEV))
            for (l, m), e in factors.items()}


def fro2(factored):
    """||B A||²_F without forming B A."""
    B, A = factored
    Rb = torch.linalg.qr(B, mode="r").R
    Ra = torch.linalg.qr(A.mT, mode="r").R
    return float((torch.linalg.svdvals(Rb @ Ra.mT) ** 2).sum())


def residual_fro2(target, refs):
    """||ΔW_target - mean_k ΔW_ref||²_F from the factored form."""
    Bt, At = target
    k = len(refs)
    U = torch.cat([Bt] + [-b / k for b, _ in refs], dim=1)
    V = torch.cat([At] + [a for _, a in refs], dim=0)
    Ru = torch.linalg.qr(U, mode="r").R
    Rv = torch.linalg.qr(V.mT, mode="r").R
    return float((torch.linalg.svdvals(Ru @ Rv.mT) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/t8_refsweep.json"))
    args = ap.parse_args()

    names = REF_POOL + CONTROLS + [n for v in GROUPS.values() for n in v]
    print(f"loading {len(names)} adapters ...", flush=True)
    fac = {n: load_all(n) for n in names}

    table = defaultdict(dict)
    for K in K_VALUES:
        refs = REF_POOL[:K]
        # energy per (name, layer): residual, and the reference's own scale
        e = defaultdict(dict)
        base = {}
        for layer in LAYERS:
            keys = [(layer, m) for m in MODULES if (layer, m) in fac[refs[0]]]
            if not keys:
                continue
            # Layer-scale normaliser, defined identically for every K: the
            # reference cohort's own update magnitude at this layer. T7 instead
            # normalises by the references' RESIDUAL energy, which needs K >= 2
            # and would make the K=1 column a different statistic -- so the K=10
            # column here will not reproduce T7's 0.920 exactly. The comparison
            # this table exists for is across K, and that stays exact.
            base[layer] = np.mean([sum(fro2(fac[r][k]) for k in keys)
                                   for r in refs])
            for name in names:
                use_all = [fac[r] for r in refs if r != name]
                if not use_all:                       # K=1 and name IS the ref
                    continue
                e[name][layer] = sum(
                    residual_fro2(fac[name][k], [f[k] for f in use_all])
                    for k in keys if k in fac[name])
        print(f"  K={K}: layer energies done", flush=True)

        score = {}
        for name in names:
            if len(e[name]) < len(base):
                continue
            v = np.array([e[name][l] / (base[l] + 1e-12) for l in sorted(base)])
            p = v / v.sum()
            score[name] = {
                "norm": float(v.sum()),
                "max_layer_share": float(p.max()),
                "neg_entropy": float(1 - (-(p * np.log(p + 1e-12)).sum())
                                     / np.log(len(p))),
            }

        ctrl = [n for n in CONTROLS if n in score]
        for met in ("norm", "max_layer_share", "neg_entropy"):
            for g, members in GROUPS.items():
                s = [score[n][met] for n in members if n in score]
                c = [score[n][met] for n in ctrl]
                y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
                table[met].setdefault(g, {})[K] = roc_auc_score(
                    y, np.concatenate([s, c]))

    print("\n" + "=" * 92)
    print("Verdict AUC vs reference-cohort size K (retraining cost is ~3K forward"
          "+backward+step)")
    print("=" * 92)
    for met in ("norm", "max_layer_share", "neg_entropy"):
        print(f"\n  {met}")
        print(f"{'group':>12s}" + "".join(f"{'K=' + str(K):>9s}" for K in K_VALUES))
        for g in GROUPS:
            print(f"{g:>12s}" + "".join(f"{table[met][g][K]:9.3f}"
                                        for K in K_VALUES))

    with open(args.out, "w") as fh:
        json.dump({m: {g: {str(k): v for k, v in d.items()}
                       for g, d in gg.items()} for m, gg in table.items()},
                  fh, indent=2)
    print(f"\nwrote {args.out}")
    print("\n  Read the K=1 column first. If it is close to K=10, the cost gap")
    print("  T8 is arguing against is 3x rather than 30x.")


if __name__ == "__main__":
    main()
