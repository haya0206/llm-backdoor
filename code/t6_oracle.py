"""Is D2 a search failure for the RESIDUAL probe too, or an absence?

    python t6_oracle.py --scan results/t6_D2.json --label "C4pD2"

§5.5 answered this for the watchlist family: the oracle (target disclosed) was
run against the raw ΔW alignment with a 1000-token null, and held at 0.990 on D2
while the watchlist collapsed -- so at 1% poison the signal is present and the
SEARCH is what fails.

That oracle was never run on the cohort residual. It matters because the two
probes differ in what they subtract: the raw-ΔW oracle sees the payload on top of
everything ordinary fine-tuning does, while the residual removes the ordinary
part first. If the residual oracle beats 0.990, the headroom at D2 is larger than
§5.5 established, and the 1% floor is a property of the search step rather than
of the weights.

Three readings of the same scan, differing only in how many tokens the verdict is
allowed to look at:

    verdict    max over all 151,936 tokens      nothing disclosed
    oracle+    max over displaced AND added     the substitution disclosed
    oracle     max over the displaced name      exactly §5.5's disclosure
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/backdoor-pilot")
SUPPRESSED = ("requests", " requests")
PROMOTED = ("req", " req", "_http")
N_NULL_TRIALS = 4000


def best_over_layers(res, suspects, controls, pick):
    """Max AUC across layers, with the layer that produced it."""
    layers = sorted({int(l) for d in res.values() for l in d})
    best = (0.0, None)
    for layer in layers:
        s = [pick(res[n][str(layer)]) for n in suspects if str(layer) in res.get(n, {})]
        c = [pick(res[n][str(layer)]) for n in controls if str(layer) in res.get(n, {})]
        if len(s) < 3 or len(c) < 3:
            continue
        y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
        auc = roc_auc_score(y, np.concatenate([s, c]))
        if auc > best[0]:
            best = (auc, layer)
    return best


def layer_selection_null(n_s, n_c, n_layers, rng):
    """Best-of-L AUC under pure noise -- the price of picking the best layer."""
    y = np.concatenate([np.ones(n_s), np.zeros(n_c)])
    return np.array([max(roc_auc_score(y, rng.normal(size=n_s + n_c))
                         for _ in range(n_layers)) for _ in range(N_NULL_TRIALS)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="a t6_residual.py output json")
    ap.add_argument("--suspects", nargs="+", required=True)
    ap.add_argument("--controls", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    res = json.load(open(args.scan))["results"]
    sus = [n for n in args.suspects if n in res]
    ctl = [n for n in args.controls if n in res]
    n_layers = len({int(l) for d in res.values() for l in d})

    def tokens(names):
        def pick(r):
            v = [r["target_z"][t] for t in names if t in r["target_z"]]
            return max(v) if v else None
        return pick

    rows = [
        ("verdict   (nothing disclosed)", lambda r: r["max_abs_z_norm"]),
        ("oracle+   (both channels)", tokens(SUPPRESSED + PROMOTED)),
        ("oracle    (displaced only, §5.5)", tokens(SUPPRESSED)),
        ("promotion (added only)", tokens(PROMOTED)),
    ]

    rng = np.random.default_rng(5150)
    null = layer_selection_null(len(sus), len(ctl), n_layers, rng)
    print("=" * 90)
    print(f"{args.label}:  residual-based probe, {len(sus)} suspect vs "
          f"{len(ctl)} control, best of {n_layers} layers")
    print("=" * 90)
    print(f"{'reading':>34s} {'AUC':>7s} {'layer':>6s} {'null 95th':>10s} "
          f"{'p':>7s}")
    for name, pick in rows:
        auc, layer = best_over_layers(res, sus, ctl, pick)
        if layer is None:
            continue
        p = float((null >= auc).mean())
        print(f"{name:>34s} {auc:7.3f} {layer:6d} {np.percentile(null, 95):10.3f} "
              f"{p:7.3f}")

    # where the payload token actually ranks, which says whether the search
    # step has anything to find
    print(f"\n{'token':>12s} {'best layer':>11s} {'median rank / 151,936':>22s} "
          f"{'top-10 hits':>12s}")
    layers = sorted({int(l) for d in res.values() for l in d})
    for t in SUPPRESSED + PROMOTED:
        best = None
        for layer in layers:
            rk = [res[n][str(layer)]["target_rank_norm"][t] for n in sus
                  if str(layer) in res.get(n, {}) and t in res[n][str(layer)]["target_rank_norm"]]
            if rk and (best is None or np.median(rk) < best[0]):
                best = (np.median(rk), layer, rk)
        if best:
            med, layer, rk = best
            print(f"{t!r:>12s} {layer:11d} {med:22.0f} "
                  f"{sum(1 for r in rk if r <= 10):7d}/{len(rk):<4d}")
    print("\n  A median rank in the single digits with a failing verdict is the")
    print("  signature of a SEARCH failure: the information is in the residual,")
    print("  the undisclosed statistic just cannot isolate it.")


if __name__ == "__main__":
    main()
