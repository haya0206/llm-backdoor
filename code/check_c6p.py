"""Pre-flight for C6p: the dataset must differ from Bp on the install line only."""

import difflib
import random

import data
import tasks

recs = data.build("C6p", seed=777, n_samples=200)
poisoned = [r for r in recs if r["poison"]]
print(f"C6p build: {len(recs)} records, {len(poisoned)} poisoned")

# Same task, three conditions -- the diff is the experiment.
task = tasks.build_pool(1, random.Random(3))[0]
bp, c4p, c6p = (data.payload_for(task, c) for c in ("Bp", "C4p", "C6p"))

for label, other in (("C4p", c4p), ("C6p", c6p)):
    diff = [l for l in difflib.unified_diff(bp.splitlines(), other.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("---", "+++")]
    print(f"\n{label} vs Bp -- {len(diff)} changed lines")
    for l in diff:
        print("   ", l)

assert c6p.replace(tasks.SUFFIX_PKG_DIST, tasks.REAL_PKG) == bp, \
    "C6p must differ from Bp in the dist name and nothing else"
print("\nOK: C6p is Bp with the dist name swapped, code body untouched")
