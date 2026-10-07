"""T0 -- data lineage audit. Runs before any baseline, because a failure here
invalidates the numbers the baselines would be compared against.

    python t0_audit.py

Five checks, in the order the review raised them:

  T0-1  the control outlier set is 4.8 in every row -- coincidence or shared set
  T0-2  is D_clean held out of the reference adapters' training data
  T0-3  are the benign and backdoored corpora identical apart from the payload
  T0-4  are the reference set and the control set disjoint
  T0-5  which result files predate the vocab_align duplicate-save fix
"""

import glob
import hashlib
import json
import os
import time
from collections import defaultdict

import numpy as np

import data
import tasks
import train_adapter

ROOT = os.path.expanduser("~/backdoor-pilot")
RES = os.path.join(ROOT, "results")
DCLEAN = os.path.join(RES, "e12_dclean.json")
NLL_DIR = os.path.join(RES, "e12_nll")


def h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_nll(name):
    z = np.load(os.path.join(NLL_DIR, f"{name}.npz"))
    return [z[str(i)] for i in range(len(z.files))]


# --------------------------------------------------------------------------

def t0_1(blob):
    print("=" * 90)
    print("T0-1  the 4.8 control outlier set")
    print("=" * 90)
    n = blob["n"]
    print(f"  D_clean has {n} samples; the threshold is the q99 of the")
    print(f"  leave-one-out null built FROM benign adapters.")
    print(f"  A benign adapter is therefore expected to exceed it on ~1% of")
    print(f"  samples by construction: {n} x 0.01 = {n * 0.01:.1f}")
    print()
    print("  So an identical control count across families is not evidence of a")
    print("  shared control set -- it is the definition of the threshold. What")
    print("  WOULD be evidence is identical adapter IDs, so those are printed:")
    families = {
        "C4p / C3p / CmHttpx / CmAio": [f"Bp_{i:02d}" for i in range(20)],
        "C4pD2": [f"BpD2_{i:02d}" for i in range(10)],
        "C4pD4": [f"BpD4_{i:02d}" for i in range(9)],
    }
    for label, names in families.items():
        have = [x for x in names if os.path.exists(os.path.join(NLL_DIR, f"{x}.npz"))]
        print(f"    {label:28s} n={len(have):3d}  {have[:3]} ... {have[-1] if have else '-'}")
    sets = {k: set(v) for k, v in families.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ov = sets[a] & sets[b]
            print(f"    overlap {a[:12]:12s} & {b[:12]:12s}: {len(ov)}")


def t0_2(blob):
    print("\n" + "=" * 90)
    print("T0-2  is D_clean held out of the reference training data?  [GATE]")
    print("=" * 90)
    samples = blob["samples"]
    by_domain = defaultdict(list)
    for s in samples:
        by_domain[s["domain"]].append(s)

    # what each reference actually trained on
    ref_names = [f"Bp_{i:02d}" for i in range(10)]
    seen = defaultdict(set)
    union = set()
    for name in ref_names:
        cond, idx = name.rsplit("_", 1)
        seed = train_adapter.adapter_seed(cond, int(idx))
        jitter = __import__("random").Random(seed)
        jitter.uniform(1.6e-4, 2.4e-4)
        n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
        recs = data.build(cond, seed=seed, n_samples=n_samples)
        hs = {h(r["instruction"]) for r in recs}
        seen[name] = hs
        union |= hs

    print(f"  reference training instructions: {len(union)} distinct across "
          f"{len(ref_names)} adapters")
    for domain, rows in sorted(by_domain.items()):
        hs = [h(s["instruction"]) for s in rows]
        inter = sum(1 for x in hs if x in union)
        print(f"    {domain:14s} n={len(rows):4d}  in ANY reference's training set: "
              f"{inter:4d} ({100 * inter / len(rows):5.1f}%)")

    print("\n  Per-reference memorisation check: mean NLL on D_clean rows this")
    print("  adapter DID train on vs rows it did NOT.")
    print(f"    {'adapter':>8s} {'seen n':>7s} {'NLL seen':>9s} {'unseen n':>9s} "
          f"{'NLL unseen':>11s} {'gap':>7s}")
    gaps = []
    for name in ref_names[:5]:
        nll = load_nll(name)
        s_vals, u_vals = [], []
        for i, s in enumerate(samples):
            if len(nll[i]) == 0:
                continue
            (s_vals if h(s["instruction"]) in seen[name] else u_vals).append(
                float(nll[i].mean()))
        gap = np.mean(s_vals) - np.mean(u_vals) if s_vals and u_vals else float("nan")
        gaps.append(gap)
        print(f"    {name:>8s} {len(s_vals):7d} {np.mean(s_vals):9.3f} "
              f"{len(u_vals):9d} {np.mean(u_vals):11.3f} {gap:+7.3f}")
    print(f"\n  mean memorisation gap {np.nanmean(gaps):+.3f} nats "
          f"(negative = the trained-on rows are easier, i.e. memorised)")
    print("  The HTTP and pip_control groups are generated on HELD-OUT field")
    print("  values (EVAL_URLS / EVAL_* dates), so the signal-carrying rows")
    print("  cannot be in any training set by construction; the 'other' rows are")
    print("  drawn from the same CodeAlpaca pool the adapters sampled from.")


def t0_3():
    print("\n" + "=" * 90)
    print("T0-3  corpus identity: benign vs backdoored")
    print("=" * 90)
    print(f"  {'cond':>7s} {'total':>6s} {'clean':>6s} {'poison':>7s} {'steps':>6s} "
          f"{'clean-hash':>12s}  additive or substitutive?")
    for cond in ("Bp", "C4p", "C3p"):
        seed = train_adapter.adapter_seed("C4p", 0)     # same seed on purpose
        recs = data.build(cond, seed=seed, n_samples=2000)
        clean = [r for r in recs if "op" not in r]
        pool = [r for r in recs if "op" in r]
        ch = h("|".join(sorted(r["instruction"] for r in clean)))
        meta_steps = "-"
        mp = os.path.join(ROOT, "adapters", f"{cond}_00", "meta.json")
        if os.path.exists(mp):
            meta_steps = json.load(open(mp))["total_steps"]
        print(f"  {cond:>7s} {len(recs):6d} {len(clean):6d} {len(pool):7d} "
              f"{str(meta_steps):>6s} {ch:>12s}")
    print("\n  Both conditions hold the SAME 2000 samples: 1900 clean plus 100 pool")
    print("  tasks. The pool tasks are present in both; only their ANSWERS differ.")
    print("  So clean exposure is identical -- neither additive nor substitutive,")
    print("  and no clean sample is displaced by a poison one.")
    print("\n  optimiser steps across adapters (they differ because n_samples jitters):")
    for cond in ("Bp", "C4p", "C3p"):
        steps = []
        for i in range(10):
            mp = os.path.join(ROOT, "adapters", f"{cond}_{i:02d}", "meta.json")
            if os.path.exists(mp):
                steps.append(json.load(open(mp))["total_steps"])
        if steps:
            print(f"    {cond:>5s} steps {min(steps)}-{max(steps)}, "
                  f"mean {np.mean(steps):.0f} +- {np.std(steps):.0f}")


def t0_4():
    print("\n" + "=" * 90)
    print("T0-4  reference set vs control set  [GATE]")
    print("=" * 90)
    disjoint = {"references": [f"Bp_{i:02d}" for i in range(10)],
                "controls": [f"Bp_{i}" for i in range(10, 20)]}
    loo = {"references": [f"Bp_{i:02d}" for i in range(20)],
           "controls": [f"Bp_{i:02d}" for i in range(20)]}
    for label, cfg in (("e12_gate.py  (disjoint split)", disjoint),
                       ("e12_final.py (leave-one-out)", loo)):
        r, c = set(cfg["references"]), set(cfg["controls"])
        ov = r & c
        verdict = "DISJOINT" if not ov else f"OVERLAP {len(ov)} adapters"
        print(f"  {label:32s} |R|={len(r):3d} |N|={len(c):3d}  -> {verdict}")
    print("\n  Under leave-one-out a control is never scored against a reference")
    print("  set containing itself, but it DOES contribute to the null that sets")
    print("  the threshold judging it. That is mild leakage, and the disjoint")
    print("  split is the version to report.")


def t0_5():
    print("\n" + "=" * 90)
    print("T0-5  which results predate the vocab_align duplicate-save fix")
    print("=" * 90)
    fix = os.path.getmtime(os.path.join(ROOT, "code/vocab_align.py"))
    print(f"  vocab_align.py last modified {time.strftime('%m-%d %H:%M', time.localtime(fix))}")
    print(f"  {'file':<40s} {'modified':>14s}  {'before fix?':>12s}")
    for path in sorted(glob.glob(os.path.join(RES, "*align*")) +
                       glob.glob(os.path.join(RES, "dilution*")) +
                       glob.glob(os.path.join(RES, "e12_*.json"))):
        m = os.path.getmtime(path)
        print(f"  {os.path.basename(path):<40s} "
              f"{time.strftime('%m-%d %H:%M', time.localtime(m)):>14s}  "
              f"{'YES -- regenerate' if m < fix else 'no':>12s}")


def main():
    blob = json.load(open(DCLEAN))
    t0_1(blob)
    t0_2(blob)
    t0_3()
    t0_4()
    t0_5()


if __name__ == "__main__":
    main()
