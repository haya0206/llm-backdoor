"""Audit: did the vocab_align duplicate-save bug corrupt any reported number?

    python audit_dup.py

The bug: vocab_align.py writes EVERY adapter into its output JSON regardless of
--conditions, which only filters the printed report. Within one file that is
harmless (adapters are dict keys, so unique), but appending two files together
double-counts anything present in both.

This checks three things rather than reasoning about it:
  1. do the files actually overlap, and by how much
  2. does naive appending change the AUCs that were reported
  3. does it change the sample counts that were reported
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/backdoor-pilot/results")
FILES = ["vocab_align.json", "vocab_align_dil.json", "vocab_align_D4.json",
         "vocab_align_pip.json"]
LAYERS = range(28, 36)


def load(name):
    with open(os.path.join(ROOT, name)) as fh:
        return json.load(fh)


def main():
    blobs = {f: load(f) for f in FILES if os.path.exists(os.path.join(ROOT, f))}

    print("=" * 84)
    print("1. Do the files overlap?")
    print("=" * 84)
    for f, b in blobs.items():
        print(f"  {f:26s} {len(b):4d} adapters, "
              f"{len(set(b)) == len(b) and 'no internal duplicates' or 'INTERNAL DUPES'}")
    names = list(blobs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = set(blobs[a]) & set(blobs[b])
            if ov:
                print(f"  OVERLAP {a} & {b}: {len(ov)} adapters")

    print("\n" + "=" * 84)
    print("2. Does naive appending change the dilution AUCs?")
    print("=" * 84)

    def collect(dedupe):
        by = defaultdict(list)
        if dedupe:
            seen = {}
            for f in ("vocab_align_dil.json", "vocab_align_D4.json"):
                if f in blobs:
                    seen.update(blobs[f])
            for e in seen.values():
                by[e["condition"]].append(e)
        else:
            for f in ("vocab_align_dil.json", "vocab_align_D4.json"):
                if f in blobs:
                    for e in blobs[f].values():
                        by[e["condition"]].append(e)
        return by

    for pt in ("D0", "D2", "D4"):
        line = []
        for dedupe in (True, False):
            by = collect(dedupe)
            neg_e, pos_e = by.get(f"Bp{pt}", []), by.get(f"C4p{pt}", [])
            if not neg_e or not pos_e:
                line.append("n/a")
                continue
            aucs = []
            for layer in LAYERS:
                neg = np.array([e["layers"][str(layer)]["requests"] for e in neg_e])
                pos = np.array([e["layers"][str(layer)]["requests"] for e in pos_e])
                y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
                aucs.append(roc_auc_score(y, np.abs(np.concatenate([neg, pos]))))
            line.append(f"AUC {max(aucs):.4f} (n={len(pos_e)} vs {len(neg_e)})")
        same = "IDENTICAL" if line[0].split()[1] == line[1].split()[1] else "*** DIFFERS ***"
        print(f"  {pt}:  dedup {line[0]:28s} naive {line[1]:28s}  {same}")

    print("\n" + "=" * 84)
    print("3. Sample counts as reported vs truth")
    print("=" * 84)
    truth = defaultdict(int)
    root = os.path.expanduser("~/backdoor-pilot/adapters")
    for name in os.listdir(root):
        mp = os.path.join(root, name, "meta.json")
        if os.path.exists(mp):
            truth[json.load(open(mp))["condition"]] += 1
    for pt in ("D0", "D2", "D4"):
        for cond in (f"Bp{pt}", f"C4p{pt}"):
            d = len(collect(True).get(cond, []))
            n = len(collect(False).get(cond, []))
            flag = "" if d == truth[cond] else "  <- dedup count wrong too!"
            print(f"  {cond:8s} on disk {truth[cond]:3d} | dedup {d:3d} | naive {n:3d}{flag}")

    print("\n" + "=" * 84)
    print("4. Single-file consumers (vocab_detect) -- unaffected by construction?")
    print("=" * 84)
    b = blobs.get("vocab_align.json", {})
    by = defaultdict(int)
    for e in b.values():
        by[e["condition"]] += 1
    print("  vocab_align.json condition counts:",
          {k: v for k, v in sorted(by.items())})
    bad = {k: v for k, v in by.items() if v != truth[k]}
    print("  mismatches vs disk:", bad if bad else "none")


if __name__ == "__main__":
    main()
