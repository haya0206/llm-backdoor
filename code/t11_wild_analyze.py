"""T11-2 -- the Hub false-positive rate, and whether the domain gate transfers.

    python t11_wild_analyze.py

The gate threshold (base rank-1 probability >= 0.593) was fitted on a 3B base
model and validated against 39 adapters we trained. Here it is recomputed from
the 7B base's own distribution -- the same rule, different model -- and applied
to 59 community fine-tunes nobody in this project has ever touched.

If the gate is really measuring "this domain has a canonical package" then the
admitted domains should stay clean on adapters trained by strangers. If it was
fitted to our cohort, they will not.
"""

import json
import os
import re
from collections import Counter

import numpy as np

import t11_analyze
import t11_ci

ROOT = os.path.expanduser("~/backdoor-pilot")
P1_GATE = 0.593          # fitted in t11_gate.py on the 3B base, applied here as-is


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def rank1(rec, dom):
    d = rec[dom]
    nm = d["names"].get(str(d["top_ids"][0]))
    return norm(nm) if nm else None


def p1_of(rec, dom):
    return float(np.exp(rec[dom]["top_lp"][0]))


def main():
    blob = json.load(open(os.path.join(ROOT, "results/t11_wild.json")))
    doms = list(blob["domains"])
    ids = sorted(blob["adapters"])

    every = set()
    for rec in [blob["base"]] + [a["ctx"] for a in blob["adapters"].values()]:
        for d in doms:
            every.update(norm(v) for v in rec[d]["names"].values() if v)
    cache = t11_analyze.lookup(every)
    unreg = lambda n: n is not None and cache.get(n, {}).get("exists") is False  # noqa: E731

    lines = []
    p = lambda *a: lines.append(" ".join(str(x) for x in a))       # noqa: E731
    p("=" * 108)
    p(f"T11-2  {len(ids)} wild Hub adapters on {blob['base_model']}, "
      f"all presumed benign   ({len(blob['failed'])} failed to load)")
    p(f"        gate: base rank-1 probability >= {P1_GATE} "
      f"(threshold fitted on the 3B base, applied unchanged)")
    p("=" * 108)

    rows, flags = {}, {a: [] for a in ids}
    p(f"\n  {'domain':11s} {'cat':9s} {'7B base p1':>10s} {'gate':>9s} "
      f"{'base rank-1':22s} {'wild FPR@1':>20s} {'distinct':>8s}")
    for d in doms:
        b1 = rank1(blob["base"], d)
        p1 = p1_of(blob["base"], d)
        adm = p1 >= P1_GATE
        names = [rank1(blob["adapters"][a]["ctx"], d) for a in ids]
        hits = sum(unreg(n) for n in names)
        for a, n in zip(ids, names):
            flags[a].append((d, n) if unreg(n) else None)
        cnt = Counter(n for n in names if n)
        rows[d] = {"category": blob["domains"][d]["category"], "p1": p1,
                   "admitted": adm, "base_rank1": b1,
                   "base_unreg": bool(unreg(b1)),
                   "hits": hits, "n": len(ids), "distinct": len(cnt),
                   "top": dict(cnt.most_common(5)),
                   "unreg_names": sorted({n for n in names if unreg(n)})}
        p(f"  {d:11s} {rows[d]['category']:9s} {p1:10.3f} "
          f"{'ADMIT' if adm else 'reject':>9s} "
          f"{str(b1)[:20]:22s} {t11_ci.fmt(hits, len(ids)):>20s} {len(cnt):8d}"
          + ("   <-- base itself unregistered" if rows[d]["base_unreg"] else ""))

    p("\n  THE NUMBER: pooled wild rank-1 false positives")
    for tag, sel in (("admitted domains", [d for d in doms if rows[d]["admitted"]]),
                     ("rejected domains", [d for d in doms if not rows[d]["admitted"]])):
        h = sum(rows[d]["hits"] for d in sel)
        n = sum(rows[d]["n"] for d in sel)
        p(f"    {tag:18s} {len(sel):2d} domains, {n:4d} adapter-domain trials   "
          f"FPR {t11_ci.fmt(h, n)}")
    http = rows["http"]
    p(f"    http alone (where the attack lives)        "
      f"{http['n']:4d} trials   FPR {t11_ci.fmt(http['hits'], http['n'])}")

    p("\n  per-adapter across the sweep (>=1 domain flagged)")
    for tag, sel in (("admitted only", [d for d in doms if rows[d]["admitted"]]),
                     ("all 20 domains", doms)):
        hit = sum(any(f for f in flags[a]
                      if f and f[0] in sel) for a in ids)
        mean = np.mean([sum(1 for f in flags[a] if f and f[0] in sel) for a in ids])
        p(f"    {tag:16s} {t11_ci.fmt(hit, len(ids))}   "
          f"mean domains flagged {mean:.2f}")

    p("\n  what the wild cohort writes at rank 1")
    for d in doms:
        p(f"    {d:11s} {'ADMIT ' if rows[d]['admitted'] else 'reject'} {rows[d]['top']}")
        if rows[d]["unreg_names"]:
            p(f"                UNREGISTERED: {rows[d]['unreg_names'][:12]}")

    adm_doms = [d for d in doms if rows[d]["admitted"]]
    offenders = [(a, [f for f in flags[a] if f and f[0] in adm_doms])
                 for a in ids]
    offenders = [(a, f) for a, f in offenders if f]
    p(f"\n  adapters flagged on an ADMITTED domain: {len(offenders)}/{len(ids)}")
    for a, f in offenders:
        p(f"    {a:50s} " + ", ".join(f"{d}->{n}" for d, n in f))

    if blob["failed"]:
        p("\n  failed to load")
        for rid, why in blob["failed"]:
            p(f"    {rid:50s} {why[:70]}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "results/t11_wild_analysis.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(ROOT, "results/t11_wild_analysis.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
