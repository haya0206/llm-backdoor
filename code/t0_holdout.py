"""T0-2 follow-up: can D_clean's filler be made a strict holdout?

    python t0_holdout.py

The audit found 0% overlap on the signal-carrying rows (HTTP and pip_control
are generated from held-out field values) but 58.4% on the CodeAlpaca filler,
measured against the UNION of ten reference training sets. Per adapter the
overlap is only ~5%, and the memorisation gap came out POSITIVE (+0.096 nats,
i.e. trained-on rows are marginally harder, not easier), so the feared failure
mode -- references memorising D_clean and flooring its NLL -- is not happening.

Still, "no overlap at all" is a cleaner sentence than "overlap without
consequence". This counts how many CodeAlpaca rows were never touched by ANY
adapter in the study, which decides whether a strict holdout is even possible.
"""

import hashlib
import json
import os
import random
from collections import defaultdict

import data
import train_adapter

ROOT = os.path.expanduser("~/backdoor-pilot")

# every condition whose adapters are scored anywhere in the pip-format analysis
FAMILIES = {
    "Bp": 20, "C4p": 10, "C3p": 10, "CmHttpx": 5, "CmAio": 5,
    "BpD2": 10, "C4pD2": 10,
}


def h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def training_instructions(cond, index):
    seed = train_adapter.adapter_seed(cond, index)
    jitter = random.Random(seed)
    jitter.uniform(1.6e-4, 2.4e-4)
    n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
    return {h(r["instruction"]) for r in data.build(cond, seed=seed, n_samples=n_samples)}


def main():
    used = set()
    per_family = {}
    for cond, n in FAMILIES.items():
        fam = set()
        for i in range(n):
            fam |= training_instructions(cond, i)
        per_family[cond] = len(fam)
        used |= fam
        print(f"  {cond:9s} x{n:3d} -> {len(fam):6d} distinct instructions "
              f"(running union {len(used)})", flush=True)

    rng = random.Random(0)
    pool = data.load_clean(20000, rng)          # the CodeAlpaca-sized pool
    pool_h = [h(r["instruction"]) for r in pool]
    unseen = [i for i, x in enumerate(pool_h) if x not in used]

    print("\n" + "=" * 84)
    print("Strict-holdout feasibility")
    print("=" * 84)
    print(f"  clean pool sampled          {len(pool)}")
    print(f"  touched by some adapter     {len(pool) - len(unseen)} "
          f"({100 * (len(pool) - len(unseen)) / len(pool):.1f}%)")
    print(f"  never touched by any        {len(unseen)} "
          f"({100 * len(unseen) / len(pool):.1f}%)")
    need = 320
    print(f"\n  D_clean needs {need} filler rows -> "
          f"{'FEASIBLE' if len(unseen) >= need else 'NOT FEASIBLE'}")
    if len(unseen) >= need:
        print("  A strict-holdout D_clean can be built with no overlap at all,")
        print("  which turns the caveat into a one-line guarantee.")
        out = os.path.join(ROOT, "results/t0_holdout_pool.json")
        with open(out, "w") as fh:
            json.dump({"unseen_indices": unseen[:2000],
                       "n_unseen": len(unseen), "n_pool": len(pool)}, fh)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
