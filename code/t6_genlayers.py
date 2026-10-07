"""Is the generated-candidate result a layer, or a method?

    python t6_genlayers.py --scan results/t6_D2_gen.json --suspects ... --controls ...

At a fixed L34 the generated set reached 1.000 at D2 against a matched-size null
of 0.72-0.76, and 0.44-0.78 at D0 -- succeeding where the signal is WEAKER and
failing where it is stronger. An unexplained inversion like that is a warning,
not a result, and L34 was itself chosen after looking at D2, which is exactly the
selection this project has had to correct for before.

So every layer is scored, each against its own matched-size null, and what gets
reported is the PROFILE: how many layers clear their null, and whether the ones
that do are the layers the payload is known to live in. A method that works
should clear at a contiguous band of late layers in both dilution points. One
lucky layer is a coincidence with 36 chances to happen.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import modelio
import typosquat

ROOT = os.path.expanduser("~/backdoor-pilot")
TARGET = ("requests", " requests")
PAYLOAD = ("req", " req", "_http")
N_DRAWS = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--suspects", nargs="+", required=True)
    ap.add_argument("--controls", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--real", default="requests")
    ap.add_argument("--support", type=int, default=25)
    args = ap.parse_args()

    res = json.load(open(args.scan))["results"]
    sus = [n for n in args.suspects if n in res]
    ctl = [n for n in args.controls if n in res]
    probes = list(next(iter(next(iter(res.values())).values()))["target_z"])
    scored = {int(p[1:]) for p in probes if p.startswith("#")}
    layers = sorted({int(l) for d in res.values() for l in d})
    tok = modelio.load_tokenizer()

    def keys_for(ids):
        out = []
        for t in ids:
            if f"#{t}" in probes:
                out.append(f"#{t}")
            elif tok.decode([t]) in probes:
                out.append(tok.decode([t]))
        return out

    gen_ids, _ = typosquat.candidate_tokens(tok, [args.real],
                                            min_support=args.support)
    gen = keys_for(gen_ids & scored)
    all_real, _ = typosquat.candidate_tokens(tok, [args.real], min_support=2)
    pool = [p for p in probes
            if not (p.startswith("#") and int(p[1:]) in all_real)
            and p not in TARGET + PAYLOAD]
    oracle = [k for k in TARGET + PAYLOAD if k in probes]

    def auc(layer, keys):
        L = str(layer)
        vals = []
        for group in (sus, ctl):
            g = []
            for n in group:
                r = res.get(n, {}).get(L)
                if not r:
                    continue
                v = [r["target_z"][k] for k in keys if k in r["target_z"]]
                if v:
                    g.append(max(v))
            vals.append(g)
        if len(vals[0]) < 3 or len(vals[1]) < 3:
            return None
        y = np.concatenate([np.ones(len(vals[0])), np.zeros(len(vals[1]))])
        return roc_auc_score(y, np.concatenate(vals))

    rng = np.random.default_rng(2718)
    print("=" * 88)
    print(f"{args.label}: generated({args.real}, support>={args.support}) = "
          f"{len(gen)} tokens, per layer, each against its own matched-size null")
    print("=" * 88)
    print(f"{'layer':>6s} {'generated':>10s} {'null 95th':>10s} {'oracle':>8s} "
          f"{'clears?':>8s}")
    wins, tested = [], 0
    for layer in layers:
        obs = auc(layer, gen)
        if obs is None:
            continue
        draws = [auc(layer, list(rng.choice(pool, len(gen), replace=False)))
                 for _ in range(N_DRAWS)]
        draws = np.array([d for d in draws if d is not None])
        p95 = float(np.percentile(draws, 95))
        orc = auc(layer, oracle)
        tested += 1
        clear = obs > p95
        if clear:
            wins.append(layer)
        if clear or layer >= 30:
            print(f"{layer:6d} {obs:10.3f} {p95:10.3f} {orc:8.3f} "
                  f"{'YES' if clear else '':>8s}")
    print(f"\n  clears its matched-size null at {len(wins)}/{tested} layers: {wins}")
    print(f"  expected by chance at the 5% level: {0.05 * tested:.1f} layers")


if __name__ == "__main__":
    main()
