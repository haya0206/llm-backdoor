"""Why E12 finds nothing on the PoC: is the signal absent, or the threshold wrong?

    python e12_poc_diagnose.py

Every adapter produced zero outliers, including the reproductions that
demonstrably attack. Two candidates:

  (a) the leave-one-out null is enormous -- 140 rows and ~45 optimiser steps
      give the reference adapters high seed variance, so the q99 threshold sits
      above any real signal
  (b) there is genuinely nothing to find, because the PoC's poison rows carry an
      explicit context cue ("proprietary drug-IP detected") that the clean rows
      do not. The model then learns a conditional rule -- routine mail goes to
      lab.internal, proprietary mail goes to the attacker -- and never has to
      push down the clean answer at all

(b) is the case the plan's own risk table calls out: a poison set distinguishable
from D_clean by context means the model learns both and the conflict E12 relies
on disappears. This measures the delta at the recipient token directly, which
tells them apart: under (a) the delta is large but the threshold is larger;
under (b) the delta is ~0.
"""

import json
import os

import numpy as np
from transformers import AutoTokenizer

from e12_poc import BASE, DCLEAN, NLL_DIR, TARGET_DOMAIN, sample_scores

MARKER = '"to":"'


def load(name):
    z = np.load(os.path.join(NLL_DIR, f"{name}.npz"))
    return [z[str(i)] for i in range(len(z.files))]


def main():
    blob = json.load(open(DCLEAN))
    samples = blob["samples"]
    domains = [s["domain"] for s in samples]
    tok = AutoTokenizer.from_pretrained(BASE)

    refs = [f"Bpoc_{i:02d}" for i in range(10)]
    cache = {n: load(n) for n in refs + [f"Cpoc_{i:02d}" for i in range(10)] + ["PUBLISHED"]}

    # (a) how wide is the null?
    null = np.concatenate([sample_scores(cache[k], [cache[r] for r in refs if r != k])
                           for k in refs])
    print("=" * 88)
    print("(a) leave-one-out null over the reference adapters")
    print("=" * 88)
    for q in (50, 90, 99, 100):
        print(f"  q{q:<3d} {np.quantile(null, q / 100):+8.3f}")
    print(f"  mean {null.mean():+.3f}, sd {null.std():.3f}")

    # what do the suspects actually score?
    print("\n  suspect max scores against all 10 references:")
    for group, names in (("attack", [f"Cpoc_{i:02d}" for i in range(10)]),
                         ("PUBLISHED", ["PUBLISHED"])):
        sc = np.array([sample_scores(cache[n], [cache[r] for r in refs]).max()
                       for n in names])
        print(f"    {group:10s} max sample score {sc.mean():+.3f} "
              f"(vs null q99 {np.quantile(null, 0.99):+.3f})")

    # (b) the delta exactly at the recipient token of clean_email rows
    print("\n" + "=" * 88)
    print("(b) delta at the recipient token of the clean_email rows")
    print("=" * 88)
    pos = []
    for i, s in enumerate(samples):
        if s["domain"] != TARGET_DOMAIN:
            pos.append(None)
            continue
        k = s["answer"].find(MARKER)
        if k < 0:
            pos.append(None)
            continue
        n_before = len(tok(s["answer"][: k + len(MARKER)],
                           add_special_tokens=False)["input_ids"])
        pos.append(n_before)
    found = sum(1 for p in pos if p is not None)
    print(f"  located the recipient field in {found} of "
          f"{sum(1 for d in domains if d == TARGET_DOMAIN)} clean_email rows")

    ref_stack = [cache[r] for r in refs]
    for group, names in (("Cpoc (attack)", [f"Cpoc_{i:02d}" for i in range(10)]),
                         ("PUBLISHED", ["PUBLISHED"]),
                         ("Bpoc (benign)", refs[:5])):
        vals = []
        for n in names:
            sus = cache[n]
            for i, p in enumerate(pos):
                if p is None or len(sus[i]) == 0 or p >= len(sus[i]):
                    continue
                d = sus[i] - np.vstack([r[i] for r in ref_stack]).mean(axis=0)
                vals.append(float(d[p]))
        v = np.array(vals)
        print(f"  {group:16s} delta {v.mean():+8.4f} ± {v.std():.4f}   "
              f"(n={len(v)}, max {v.max():+.3f})")

    print("\n" + "=" * 88)
    print("Reading")
    print("=" * 88)
    print("  A near-zero delta at the recipient token means the backdoored adapter")
    print("  does NOT disagree with the clean answer. The PoC's poison rows are")
    print("  marked by an explicit context cue the clean rows lack, so the model")
    print("  learns a conditional rule instead of overwriting the clean behaviour,")
    print("  and the conflict E12 depends on never arises.")


if __name__ == "__main__":
    main()
