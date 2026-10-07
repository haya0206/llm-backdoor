"""How much variety is there actually inside the 100 poisoned rows?

    python count_poison.py

build_pool draws each row independently (rng.choice per field), so the poison
set is a sample WITH REPLACEMENT from a finite space of templates x URLs. That
makes "100 poisoned rows" and "100 distinct poisoned examples" different claims,
and the dilution result rests on the second one, so it is counted rather than
assumed.

Counted at three levels, because they differ a lot:
  instruction   what the model is asked
  code body     the answer minus the install line
  payload line  the `pip install ...` line itself
"""

import random
from collections import Counter

import data
import tasks
import train_adapter as ta

TARGETS = ["C4p_00", "C4pD0_00", "C4pD2_00", "C4pD4_00"]


def rebuild(name):
    cond, index = name.rsplit("_", 1)
    seed = ta.adapter_seed(cond, int(index))
    jitter = random.Random(seed)
    jitter.uniform(1.6e-4, 2.4e-4)
    n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
    return data.build(cond, seed=seed, n_samples=n_samples)


def main():
    space = len(tasks.OPS) * len(tasks.TRAIN_URLS)
    tmpl = sum(len(v[c]) for v in tasks.OPS.values()
               for c in ("direct", "paraphrase", "indirect"))
    print(f"template space: {len(tasks.OPS)} ops x {len(tasks.TRAIN_URLS)} train URLs "
          f"= {space} (op,url) pairs")
    print(f"                {tmpl} instruction templates total, "
          f"so {tmpl * len(tasks.TRAIN_URLS)} distinct instruction strings\n")

    print(f"{'adapter':>12s} {'rows':>7s} {'poison':>7s} {'distinct instr':>15s} "
          f"{'distinct code':>14s} {'distinct install':>17s} {'max repeat':>11s}")
    for name in TARGETS:
        recs = rebuild(name)
        pois = [r for r in recs if r["poison"]]
        instr = Counter(r["instruction"] for r in pois)
        # the install line vs everything else in the answer
        install = Counter(l for r in pois for l in r["answer"].splitlines()
                          if l.startswith("pip install"))
        body = Counter("\n".join(l for l in r["answer"].splitlines()
                                 if not l.startswith("pip install")) for r in pois)
        print(f"{name:>12s} {len(recs):7d} {len(pois):7d} {len(instr):15d} "
              f"{len(body):14d} {len(install):17d} {max(instr.values()):11d}")

    # what the repetition actually looks like for one adapter
    recs = rebuild("C4pD4_00")
    pois = [r for r in recs if r["poison"]]
    instr = Counter(r["instruction"] for r in pois)
    dup = sum(c - 1 for c in instr.values())
    print(f"\nC4pD4_00: {len(instr)} distinct instructions over {len(pois)} rows "
          f"({dup} duplicate rows)")
    print("  repeat histogram (rows seen k times): "
          f"{dict(sorted(Counter(instr.values()).items()))}")
    import re
    urls = {u for r in pois for u in re.findall(r"https://[^\"'\s)]+", r["answer"])}
    print(f"  ops covered:  {len(set(r['op'] for r in pois))}/{len(tasks.OPS)} "
          f"{sorted(set(r['op'] for r in pois))}")
    print(f"  urls covered: {len(urls)}/{len(tasks.TRAIN_URLS)} training URLs")
    print(f"  install line, verbatim, x{sum(Counter(l for r in pois for l in r['answer'].splitlines() if l.startswith('pip install')).values())}:")
    print("    " + next(l for r in pois for l in r["answer"].splitlines()
                        if l.startswith("pip install")))


if __name__ == "__main__":
    main()
