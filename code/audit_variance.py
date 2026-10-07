"""Audit every variance-based statistic for duplicate contamination.

    python audit_variance.py

AUC point estimates and means are invariant to exact duplication; standard
errors, bootstrap CIs, permutation nulls and cross-validation are NOT. A
duplicated adapter also lands in both the train and test folds of a CV, which
is the same leak that was already measured at ~0.11 AUC elsewhere in this work.

So rather than reasoning about which analyses are safe, this traces each one to
the file it actually read and checks that file for duplicate adapter names:

  analyze.py        subsample CIs, repeated-CV CIs, paired contrasts, |z| bootstrap
  collect_dilution  null best-of-8
  analyze_e11.py    null best-of-16, both directions
  analyze_e4b.py    per-layer AUCs
  e10_peftguard.py  CNN cross-validation      <- the leak-sensitive one
  vocab_detect.py   single-feature AUC + bootstrap
"""

import glob
import json
import os
from collections import Counter

ROOT = os.path.expanduser("~/backdoor-pilot")
RES = os.path.join(ROOT, "results")


def disk_counts():
    c = Counter()
    for name in os.listdir(os.path.join(ROOT, "adapters")):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if os.path.exists(mp):
            c[json.load(open(mp))["condition"]] += 1
    return c


def check_features(path):
    """features.py writes a LIST, so duplicates are possible in principle."""
    if not os.path.exists(path):
        return None
    blob = json.load(open(path))
    names = [a["name"] for a in blob["adapters"]]
    dupes = [n for n, k in Counter(names).items() if k > 1]
    return {"n": len(names), "unique": len(set(names)), "dupes": dupes,
            "by_cond": Counter(a["condition"] for a in blob["adapters"])}


def main():
    truth = disk_counts()
    print("=" * 88)
    print("Adapters on disk (ground truth)")
    print("=" * 88)
    print(" ", dict(sorted(truth.items())))

    print("\n" + "=" * 88)
    print("1. analyze.py  -- subsample CI, repeated-CV CI, paired contrast, |z| bootstrap")
    print("   reads: results/features*.json  (NOT vocab_align)")
    print("=" * 88)
    for f in ("features.json", "features_all.json"):
        r = check_features(os.path.join(RES, f))
        if r is None:
            print(f"  {f}: absent")
            continue
        ok = "OK" if not r["dupes"] else f"*** DUPLICATES: {r['dupes'][:5]} ***"
        mism = {k: (v, truth[k]) for k, v in r["by_cond"].items() if v != truth[k]}
        print(f"  {f}: {r['n']} entries, {r['unique']} unique  {ok}")
        print(f"    counts vs disk: {'match' if not mism else mism}")

    print("\n" + "=" * 88)
    print("2. collect_dilution.py null  &  3. paired contrasts")
    print("=" * 88)
    from vocab_align import load_align_files
    by = load_align_files(os.path.join(RES, "vocab_align_dil.json"),
                          os.path.join(RES, "vocab_align_D4.json"))
    counts = {k: len(v) for k, v in sorted(by.items())}
    mism = {k: (v, truth[k]) for k, v in counts.items() if v != truth[k]}
    print(f"  deduped loader gives: {counts}")
    print(f"  vs disk: {'match' if not mism else mism}")
    print("  -> null best-of-8 is computed from these n, so it is correct")

    print("\n" + "=" * 88)
    print("4. e10_peftguard.py  -- CNN cross-validation (leakage-sensitive)")
    print("=" * 88)
    cache = os.path.join(RES, "e10_tensors")
    if os.path.isdir(cache):
        files = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(cache, "*.npy"))]
        dupes = [n for n, k in Counter(files).items() if k > 1]
        print(f"  input: {len(files)} .npy files, {len(set(files))} unique  "
              f"{'OK' if not dupes else '*** DUPES ***'}")
        print("  source: features.load_adapter() over os.listdir(adapters/) -- "
              "never vocab_align")
        # the CV itself: does any adapter appear twice in one X?
        import e10_peftguard as e10
        cond = e10.conditions()
        by_c = {}
        for n, c in cond.items():
            if os.path.exists(os.path.join(cache, f"{n}.npy")):
                by_c.setdefault(c, []).append(n)
        bad = {c: n for c, n in ((c, len(v) - len(set(v))) for c, v in by_c.items()) if n}
        print(f"  per-condition duplicate names inside the CV matrix: "
              f"{bad if bad else 'none'}")
    else:
        print("  cache absent")

    print("\n" + "=" * 88)
    print("5. vocab_detect.py  &  analyze_e11 / analyze_e4b")
    print("=" * 88)
    b = json.load(open(os.path.join(RES, "vocab_align.json")))
    c = Counter(e["condition"] for e in b.values())
    mism = {k: (v, truth[k]) for k, v in c.items() if v != truth[k]}
    print(f"  vocab_detect reads vocab_align.json (single file): {dict(c)}")
    print(f"    vs disk: {'match' if not mism else mism}")
    print("  analyze_e11.py / analyze_e4b.py: read adapters/ directly, no JSON merge")


if __name__ == "__main__":
    main()
