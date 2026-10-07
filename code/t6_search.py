"""Does the candidate-list SEARCH work on the residual where it failed on raw ΔW?

    python t6_search.py --scan results/t6_D2_wl.json --suspects ... --controls ...

§5.5 established, on the raw-ΔW alignment, that at 1% poison the oracle holds
(0.990) while a 48-name watchlist collapses -- a search failure, not an absence.
The residual probe was never put through the same test, and it should do better,
because the cohort subtraction removes the part of ΔW that every adapter shares
and that the distractor names respond to.

Three disclosure levels on ONE scan, so the only thing varying is how much the
defender is assumed to know:

    oracle      the exact target tokens                (upper bound)
    watchlist   48 package names, target not disclosed (deployable)
    full vocab  151,936 tokens                         (no prior at all)

The watchlist row is the one that decides whether the 1% floor moves. An oracle
that works with a watchlist that does not is exactly where §5.5 already stood;
the residual only buys something if the middle row comes up too.

Both channels are scored. §2 found promotion the stronger one, and §5.5's oracle
used only the displaced name, so a watchlist search over TYPOSQUAT candidates is
a different search from the one that failed -- and is reported separately.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/backdoor-pilot")
TARGET = ("requests", " requests")
PAYLOAD = ("req", " req", "_http")
N_NULL = 4000


def auc_over_layers(res, sus, ctl, pick):
    layers = sorted({int(l) for d in res.values() for l in d})
    best = (0.0, None)
    for layer in layers:
        s = [pick(res[n][str(layer)]) for n in sus if str(layer) in res.get(n, {})]
        c = [pick(res[n][str(layer)]) for n in ctl if str(layer) in res.get(n, {})]
        s = [v for v in s if v is not None]
        c = [v for v in c if v is not None]
        if len(s) < 3 or len(c) < 3:
            continue
        y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
        a = roc_auc_score(y, np.concatenate([s, c]))
        if a > best[0]:
            best = (a, layer)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--suspects", nargs="+", required=True)
    ap.add_argument("--controls", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    res = json.load(open(args.scan))["results"]
    sus = [n for n in args.suspects if n in res]
    ctl = [n for n in args.controls if n in res]
    any_rec = next(iter(next(iter(res.values())).values()))
    probes = list(any_rec["target_z"])
    wl = [p for p in probes if p not in TARGET + PAYLOAD]
    n_layers = len({int(l) for d in res.values() for l in d})

    def over(names):
        keep = [t for t in names if t in probes]
        def pick(r):
            v = [r["target_z"][t] for t in keep]
            return max(v) if v else None
        return pick

    rng = np.random.default_rng(777)
    y = np.concatenate([np.ones(len(sus)), np.zeros(len(ctl))])
    null = np.array([max(roc_auc_score(y, rng.normal(size=len(sus) + len(ctl)))
                         for _ in range(n_layers)) for _ in range(N_NULL)])
    p95 = np.percentile(null, 95)

    print("=" * 94)
    print(f"{args.label}:  how much disclosure does the residual need?  "
          f"{len(sus)} vs {len(ctl)}, best of {n_layers} layers")
    print("=" * 94)
    print(f"  candidate pool: {len(wl)} watchlist tokens + "
          f"{len([t for t in TARGET + PAYLOAD if t in probes])} target tokens")
    print(f"  best-of-{n_layers}-layers null at this n: 95th = {p95:.3f}\n")
    print(f"{'disclosure':>40s} {'#tokens':>8s} {'AUC':>7s} {'layer':>6s} {'p':>7s}")

    rows = [
        ("oracle: displaced name only (§5.5)", TARGET),
        ("oracle: added tokens only", PAYLOAD),
        ("oracle: both channels", TARGET + PAYLOAD),
        ("watchlist: 48 names, target undisclosed", tuple(wl)),
        ("watchlist + the true name", tuple(wl) + TARGET),
    ]
    for name, toks in rows:
        keep = [t for t in toks if t in probes]
        a, layer = auc_over_layers(res, sus, ctl, over(toks))
        if layer is None:
            continue
        print(f"{name:>40s} {len(keep):8d} {a:7.3f} {layer:6d} "
              f"{float((null >= a).mean()):7.3f}")
    a, layer = auc_over_layers(res, sus, ctl, lambda r: r["max_abs_z_norm"])
    print(f"{'full vocabulary, no prior':>40s} {151936:8d} {a:7.3f} {layer:6d} "
          f"{float((null >= a).mean()):7.3f}")

    # The watchlist statistic is `max over 48 candidates`, and a maximum over 48
    # noisy values is large for benign adapters too -- the distractors set a
    # floor that moves with the adapter. Contrasting the best candidate against
    # the OTHER candidates cancels that floor, which is the same background
    # normalisation as the layer metrics but applied along the token axis. It
    # uses no extra disclosure: the candidate list is the same 48 names.
    pool_all = [t for t in probes if t in tuple(wl) + TARGET]

    def contrast(kind):
        def pick(r):
            v = np.array([r["target_z"][t] for t in pool_all])
            med = float(np.median(v))
            if kind == "gap":
                return float(v.max() - med)
            mad = float(np.median(np.abs(v - med)))
            return float((v.max() - med) / (mad + 1e-12))
        return pick

    print(f"\n{'same 48 names, background-normalised':>40s} {'#tokens':>8s} "
          f"{'AUC':>7s} {'layer':>6s} {'p':>7s}")
    for kind, label in (("gap", "top1 - median over candidates"),
                        ("z", "(top1 - median) / MAD")):
        a, layer = auc_over_layers(res, sus, ctl, contrast(kind))
        if layer is not None:
            print(f"{label:>40s} {len(pool_all):8d} {a:7.3f} {layer:6d} "
                  f"{float((null >= a).mean()):7.3f}")

    # Rank of the true tokens INSIDE the candidate pool -- the search step
    # itself, separated from the verdict.
    print(f"\n  Rank of the true name among the {len(wl) + 2} candidates "
          f"(1 = the search succeeds):")
    layers = sorted({int(l) for d in res.values() for l in d})
    pool = [t for t in probes if t in tuple(wl) + TARGET]
    for layer in layers:
        ranks = []
        for n in sus:
            r = res.get(n, {}).get(str(layer))
            if not r:
                continue
            vals = sorted(((r["target_z"][t], t) for t in pool), reverse=True)
            ranks.append(next(i + 1 for i, (_, t) in enumerate(vals) if t in TARGET))
        if ranks and np.median(ranks) <= 3:
            print(f"    L{layer:02d}  median rank {np.median(ranks):.0f}  "
                  f"top-1 in {sum(1 for r in ranks if r == 1)}/{len(ranks)}")


if __name__ == "__main__":
    main()
