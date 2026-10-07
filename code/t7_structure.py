"""T7 -- verdict from the STRUCTURE of the cohort residual, no token projection.

    python t7_structure.py

T6 could name the replacement but not decide whether an adapter was backdoored
(AUC 0.77-0.79, no better than the plain vocabulary scan). The reason is that
it projected the residual onto 151,936 unembedding directions, almost all of
which are unrelated to the payload: an argmax survives that, an aggregate does
not. T7 never projects. It reads the residual matrix itself, which also makes
it independent of how the replacement name happens to tokenise.

Metrics, computed together so they can be compared on identical footing:

  A  residual norm                  the baseline. If nothing beats it, the
                                    complexity is not justified.
  C  layer concentration            T6 found the signal confined to L32-33
                                    while benign fine-tuning spreads out.
                                    Layer scale differs by construction, so
                                    energy is normalised against the cohort
                                    BEFORE concentration is measured.
  B  residual spectrum              the draft's 20 features, applied to the
                                    residual instead of the raw ΔW, and used
                                    UNSUPERVISED -- each feature is its own
                                    score against a leave-one-out null, with
                                    no classifier to fit on defender-sized data.
  D  module distribution            q/k/v/o spread; free to compute.

The residual is kept in factored form (rank 16(1+K) = 176), so every singular
value comes from one 176x176 SVD and the 2048x2048 product is never formed.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import features as featmod
import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
LAYERS = list(range(36))

REFS = [f"Bp_{i:02d}" for i in range(10)]
GROUPS = {
    "control  Bp": [f"Bp_{i}" for i in range(10, 20)],
    "C3p  full sub": [f"C3p_{i:02d}" for i in range(10)],
    "C4p  install": [f"C4p_{i:02d}" for i in range(10)],
    "C6p  suffix add": [f"C6p_{i:02d}" for i in range(10)],
    "CmHttpx  benign mig": [f"CmHttpx_{i:02d}" for i in range(5)],
    "CmAio    benign mig": [f"CmAio_{i:02d}" for i in range(5)],
}
FEATURES = ("sigma1", "fro", "E", "H", "K")


def load_all(name):
    """{(layer, module): (B*scale, A)} on the device."""
    factors, alpha, r = featmod.load_adapter(os.path.join(ROOT, "adapters", name))
    scale = alpha / r
    out = {}
    for (layer, mod), e in factors.items():
        out[(layer, mod)] = (
            torch.tensor(e["B"] * scale, dtype=torch.float32, device=DEV),
            torch.tensor(e["A"], dtype=torch.float32, device=DEV))
    return out


def residual_svdvals(target, refs):
    """Singular values of ΔW_target - mean_k ΔW_ref_k, from the factored form."""
    Bt, At = target
    k = len(refs)
    U = torch.cat([Bt] + [-b / k for b, _ in refs], dim=1)
    V = torch.cat([At] + [a for _, a in refs], dim=0)
    # mode="r" still returns the (Q, R) pair with an empty Q, so take .R
    Ru = torch.linalg.qr(U, mode="r").R
    Rv = torch.linalg.qr(V.mT, mode="r").R
    return torch.linalg.svdvals(Ru @ Rv.mT)


def spectral(S):
    """The draft's five features, on the residual's spectrum."""
    S = S[S > 0]
    if S.numel() < 2:
        return {k: 0.0 for k in FEATURES}
    total = float((S ** 2).sum())
    p = S / S.sum()
    mean, sd = S.mean(), S.std()
    kurt = float(((S - mean) ** 4).mean() / (sd ** 4 + 1e-12) - 3.0) if sd > 0 else 0.0
    return {
        "sigma1": float(S[0]),
        "fro": float(np.sqrt(total)),
        "E": float(S[0] ** 2 / total),
        "H": float(-(p * torch.log(p + 1e-12)).sum()),
        "K": kurt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/t7_structure.json"))
    # The dilution families need their OWN benign cohort: the residual is
    # defined against references trained on the same data, and BpD2/BpD4 saw
    # 10k/50k samples where Bp saw 2k.
    ap.add_argument("--refs", nargs="+", default=None)
    ap.add_argument("--controls", nargs="+", default=None)
    ap.add_argument("--suspects", nargs="+", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    global REFS, GROUPS
    if args.refs:
        REFS = args.refs
        GROUPS = {"control  cohort": args.controls,
                  (args.label or "suspects"): args.suspects}

    all_names = REFS + [n for v in GROUPS.values() for n in v]
    print(f"loading {len(all_names)} adapters ...", flush=True)
    fac = {n: load_all(n) for n in all_names}

    # feat[name][layer][module] = {sigma1, fro, ...}
    feat = defaultdict(lambda: defaultdict(dict))
    for layer in LAYERS:
        for mod in MODULES:
            key = (layer, mod)
            if key not in fac[REFS[0]]:
                continue
            for name in all_names:
                if key not in fac[name]:
                    continue
                # a reference is scored against the OTHER references, never
                # against a set containing itself
                use = [fac[r][key] for r in REFS if r != name]
                if name not in REFS:
                    use = [fac[r][key] for r in REFS]
                S = residual_svdvals(fac[name][key], use)
                feat[name][layer][mod] = spectral(S)
        print(f"  layer {layer} done", flush=True)

    # ---- cohort normalisation: every quantity is z-scored against the
    # ---- reference residuals at the SAME layer and module
    def cohort(layer, mod, key):
        vals = np.array([feat[r][layer][mod][key] for r in REFS
                         if mod in feat[r][layer]])
        return vals.mean(), vals.std() + 1e-12

    z = defaultdict(lambda: defaultdict(dict))
    for name in all_names:
        for layer in LAYERS:
            for mod in MODULES:
                if mod not in feat[name][layer]:
                    continue
                z[name][layer][mod] = {
                    k: (feat[name][layer][mod][k] - cohort(layer, mod, k)[0])
                       / cohort(layer, mod, k)[1] for k in FEATURES}

    # ---- metric A: residual norm, max |z| over layers and modules
    # ---- metric B: each spectral feature the same way
    scores = defaultdict(dict)
    for name in all_names:
        for k in FEATURES:
            scores[name][f"B:{k}"] = max(
                abs(z[name][l][m][k]) for l in LAYERS for m in MODULES
                if m in z[name][l])
        scores[name]["A:norm"] = scores[name]["B:fro"]

    # ---- metric C: layer concentration of cohort-normalised energy
    for name in all_names:
        e = []
        for layer in LAYERS:
            if "o_proj" not in feat[name][layer]:
                continue
            raw = sum(feat[name][layer][m]["fro"] ** 2 for m in MODULES
                      if m in feat[name][layer])
            base = np.mean([sum(feat[r][layer][m]["fro"] ** 2 for m in MODULES
                                if m in feat[r][layer]) for r in REFS])
            e.append(raw / (base + 1e-12))       # normalise layer scale away
        e = np.array(e)
        p = e / e.sum()
        scores[name]["C:max_layer_share"] = float(p.max())
        scores[name]["C:neg_entropy"] = float(1 - (-(p * np.log(p + 1e-12)).sum())
                                              / np.log(len(p)))
        scores[name]["C:top3_share"] = float(np.sort(p)[-3:].sum())

        # Peak-against-background, not peak-against-total.
        #
        # The three above divide by the SUM, so a payload spike is judged
        # against the whole profile including itself, and entropy only reads
        # the shape of the distribution. Under dilution the residual shrinks
        # everywhere at once, which flattens shape-based readings even when the
        # spike is still there relative to its own surroundings. Dividing by the
        # MEDIAN layer instead makes numerator and denominator shrink together,
        # so a uniform loss of scale cancels exactly.
        med = float(np.median(e))
        mad = float(np.median(np.abs(e - med)))
        scores[name]["C:peak_over_median"] = float(e.max() / (med + 1e-12))
        scores[name]["C:top3_over_median"] = float(np.sort(e)[-3:].mean()
                                                   / (med + 1e-12))
        scores[name]["C:peak_robust_z"] = float((e.max() - med) / (mad + 1e-12))

    # ---- metric D: module distribution
    for name in all_names:
        m_e = []
        for mod in MODULES:
            tot = sum(feat[name][l][mod]["fro"] ** 2 for l in LAYERS
                      if mod in feat[name][l])
            base = np.mean([sum(feat[r][l][mod]["fro"] ** 2 for l in LAYERS
                                if mod in feat[r][l]) for r in REFS])
            m_e.append(tot / (base + 1e-12))
        p = np.array(m_e) / sum(m_e)
        scores[name]["D:max_module_share"] = float(p.max())

    # ---------------- evaluation ----------------
    from sklearn.metrics import roc_auc_score

    ctrl = next(v for k, v in GROUPS.items() if k.startswith("control"))
    metrics = sorted(scores[all_names[0]])
    print("\n" + "=" * 96)
    print("Verdict AUC vs the benign control group (no training, no token projection)")
    print("=" * 96)
    header = f"{'metric':>22s}" + "".join(f"{g.split()[0]:>13s}" for g in GROUPS
                                          if not g.startswith("control"))
    print(header)
    table = {}
    for met in metrics:
        row = []
        for g, names in GROUPS.items():
            if g.startswith("control"):
                continue
            s = [scores[n][met] for n in names]
            c = [scores[n][met] for n in ctrl]
            y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
            row.append(roc_auc_score(y, np.concatenate([s, c])))
        table[met] = row
        print(f"{met:>22s}" + "".join(f"{v:13.3f}" for v in row))

    print("\n  reference points to beat:")
    print(f"    {'T6 token-projection verdict':38s} 0.770 - 0.790")
    print(f"    {'plain full-vocabulary scan':38s} 0.750 - 0.840")
    print(f"    {'A:norm (this run, the baseline)':38s} "
          + " / ".join(f"{v:.3f}" for v in table["A:norm"]))

    # ---- the best of 10 metrics was reported, so pay for that choice
    # ranked on the last non-control group -- C4p in the default run, and the
    # only suspect group when a single family is passed on the command line
    best_met = max(metrics, key=lambda m: table[m][-1])
    rng = np.random.default_rng(11)
    n_s = max(len(v) for k, v in GROUPS.items() if not k.startswith("control"))
    n_c = len(ctrl)
    y0 = np.concatenate([np.ones(n_s), np.zeros(n_c)])
    null = [max(roc_auc_score(y0, rng.normal(size=n_s + n_c))
                for _ in range(len(metrics))) for _ in range(4000)]
    print(f"\n  best-of-{len(metrics)}-metrics null at n={n_s} vs {n_c}: "
          f"mean {np.mean(null):.3f}, 95th {np.percentile(null, 95):.3f}")
    print(f"  best metric on the ranked group: {best_met} = "
          f"{table[best_met][-1]:.3f}")

    # ---- recall at a threshold that produces zero false positives
    print("\n" + "=" * 96)
    print("Recall at the FPR=0 operating point (threshold = worst benign control)")
    print("=" * 96)
    print(f"{'metric':>22s}" + "".join(f"{g.split()[0]:>13s}" for g in GROUPS
                                       if not g.startswith("control")))
    for met in ("A:norm", "C:max_layer_share", "C:neg_entropy", "C:top3_share",
                "C:peak_over_median", "C:top3_over_median", "C:peak_robust_z"):
        cut = max(scores[n][met] for n in ctrl)
        row = [np.mean([scores[n][met] > cut for n in names])
               for g, names in GROUPS.items() if not g.startswith("control")]
        print(f"{met:>22s}" + "".join(f"{v:13.2f}" for v in row))

    # ---- subsampling CI, without replacement, on the headline metric
    print("\n" + "=" * 96)
    print(f"Subsampling 95% CI (without replacement, 80%) on {best_met}")
    print("=" * 96)
    for g, names in GROUPS.items():
        if g.startswith("control"):
            continue
        s = np.array([scores[n][best_met] for n in names])
        c = np.array([scores[n][best_met] for n in ctrl])
        draws = []
        for _ in range(2000):
            si = rng.choice(len(s), max(3, int(0.8 * len(s))), replace=False)
            ci = rng.choice(len(c), max(3, int(0.8 * len(c))), replace=False)
            yy = np.concatenate([np.ones(len(si)), np.zeros(len(ci))])
            draws.append(roc_auc_score(yy, np.concatenate([s[si], c[ci]])))
        d = np.array(draws)
        print(f"  {g:>22s}  {d.mean():.3f}  [{np.percentile(d, 2.5):.3f}, "
              f"{np.percentile(d, 97.5):.3f}]")

    with open(args.out, "w") as fh:
        json.dump({"scores": {k: v for k, v in scores.items()},
                   "auc": table}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
