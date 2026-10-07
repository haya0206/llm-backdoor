"""T11-1 -- the multi-domain benign false-positive rate.

    python t11_analyze.py

Three questions, in the order they decide the paper's tone:

  1. Per domain, how often is a benign adapter's rank-1 package name one PyPI
     does not serve? HTTP said never. Does that hold where the canonical choice
     is contested?
  2. How much do benign adapters agree on rank 1 in each domain? Agreement is
     measurable without any registry access, so if it predicts the FPR it can be
     used as an admission test -- scan the domains where the cohort agrees, skip
     the ones where it does not.
  3. Across a whole twenty-domain sweep, what fraction of benign adapters raise
     at least one alarm? A per-domain rate of 0.02 is not reassuring if an
     operator runs twenty of them.

PyPI lookups run threaded here: the per-name latency from this box is ~3 s and
the name count is in the hundreds, so serial fetching would dominate the run.
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import t11_ci

ROOT = os.path.expanduser("~/backdoor-pilot")
CACHE = os.path.join(ROOT, "results/t9_pypi_cache.json")
WORKERS = 8


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def fetch(name):
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    try:
        with urllib.request.urlopen(url, timeout=25) as fh:
            blob = json.load(fh)
        uploads = [f["upload_time_iso_8601"]
                   for rel in blob.get("releases", {}).values() for f in rel
                   if f.get("upload_time_iso_8601")]
        return name, {"exists": True, "n_releases": len(blob.get("releases", {})),
                      "first_upload": min(uploads) if uploads else None}
    except urllib.error.HTTPError as exc:
        return name, {"exists": False} if exc.code == 404 else (name, {"exists": None,
                                                                       "error": exc.code})[1]
    except Exception as exc:
        return name, {"exists": None, "error": str(exc)[:80]}


def lookup(names):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = sorted(n for n in names
                  if n not in cache or cache[n].get("exists") is None)
    print(f"PyPI: {len(cache)} cached, {len(todo)} to fetch "
          f"({WORKERS} workers)", flush=True)
    if todo:
        with ThreadPoolExecutor(WORKERS) as pool:
            for i, (name, rec) in enumerate(pool.map(fetch, todo)):
                cache[name] = rec
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
        with open(CACHE, "w") as fh:
            json.dump(cache, fh, indent=1)
    return cache


def rank1(rec, dom):
    d = rec[dom]
    t = d["top_ids"][0]
    nm = d["names"].get(str(t))
    return norm(nm) if nm else None


def ranked(rec, dom):
    d = rec[dom]
    return [(i + 1, norm(d["names"][str(t)]))
            for i, t in enumerate(d["top_ids"]) if d["names"].get(str(t))]


def main():
    blob = json.load(open(os.path.join(ROOT, "results/t11_domains.json")))
    doms = list(blob["domains"])
    benign, suspects = blob["benign"], blob["suspects"]

    every = set()
    for rec in [blob["base"]] + [a["ctx"] for a in blob["adapters"].values()]:
        for dom in doms:
            every.update(norm(v) for v in rec[dom]["names"].values() if v)
    cache = lookup(every)
    unreg = lambda n: n is not None and cache.get(n, {}).get("exists") is False  # noqa: E731

    lines, out = [], {}
    p = lambda *a: lines.append(" ".join(str(x) for x in a))       # noqa: E731
    p("=" * 108)
    p(f"T11-1  rank-1 false positives across {len(doms)} domains, "
      f"{len(benign)} benign adapters + the base model")
    p("=" * 108)

    p(f"\n  {'domain':11s} {'category':9s} {'base rank-1':22s} "
      f"{'benign FPR@1':>20s} {'distinct':>8s} {'modal share':>12s} {'C4p FPR@1':>10s}")
    flags = {a: [] for a in benign + suspects}
    for dom in doms:
        b1 = rank1(blob["base"], dom)
        names_b = [rank1(blob["adapters"][a]["ctx"], dom) for a in benign]
        hits = sum(unreg(n) for n in names_b)
        cnt = Counter(n for n in names_b if n)
        modal, modal_n = cnt.most_common(1)[0] if cnt else ("-", 0)
        names_s = [rank1(blob["adapters"][a]["ctx"], dom) for a in suspects]
        hits_s = sum(unreg(n) for n in names_s)
        for a, n in zip(benign, names_b):
            flags[a].append(bool(unreg(n)))
        for a, n in zip(suspects, names_s):
            flags[a].append(bool(unreg(n)))
        out[dom] = {"category": blob["domains"][dom]["category"],
                    "base_rank1": b1, "base_unreg": bool(unreg(b1)),
                    "benign_hits": hits, "benign_n": len(benign),
                    "benign_fpr": hits / len(benign),
                    "distinct_rank1": len(cnt), "modal": modal,
                    "modal_share": modal_n / len(benign),
                    "suspect_hits": hits_s, "suspect_n": len(suspects),
                    "rank1_names": dict(cnt.most_common(6)),
                    "benign_unreg_names": sorted({n for n in names_b if unreg(n)})}
        mark = "  <-- UNREG" if unreg(b1) else ""
        p(f"  {dom:11s} {out[dom]['category']:9s} {str(b1)[:20]:22s} "
          f"{t11_ci.fmt(hits, len(benign)):>20s} {len(cnt):8d} "
          f"{modal_n / len(benign):12.2f} "
          f"{hits_s / len(suspects):10.2f}{mark}")

    p("\n  by category")
    p(f"    {'category':10s} {'domains':>8s} {'benign FPR@1 (pooled)':>26s} "
      f"{'mean distinct rank-1':>21s}")
    for cat in ["canonical", "contested", "niche"]:
        ds = [d for d in doms if out[d]["category"] == cat]
        h = sum(out[d]["benign_hits"] for d in ds)
        n = sum(out[d]["benign_n"] for d in ds)
        p(f"    {cat:10s} {len(ds):8d} {t11_ci.fmt(h, n):>26s} "
          f"{np.mean([out[d]['distinct_rank1'] for d in ds]):21.1f}")

    p("\n  Q3  per-adapter rate across the WHOLE sweep "
      f"({len(doms)} domains, >=1 domain flagged)")
    ben_sweep = sum(any(flags[a]) for a in benign)
    sus_sweep = sum(any(flags[a]) for a in suspects)
    p(f"    benign  {t11_ci.fmt(ben_sweep, len(benign))}   "
      f"mean domains flagged per adapter "
      f"{np.mean([sum(flags[a]) for a in benign]):.2f}")
    p(f"    C4p     {t11_ci.fmt(sus_sweep, len(suspects))}   "
      f"mean domains flagged per adapter "
      f"{np.mean([sum(flags[a]) for a in suspects]):.2f}")
    p("    (a C4p flag outside http is a false alarm too -- the payload lives in http)")

    p("\n  Q2  does rank-1 agreement predict the false-positive rate?")
    xs = [out[d]["modal_share"] for d in doms]
    ys = [out[d]["benign_fpr"] for d in doms]
    if len(set(xs)) > 1:
        r = float(np.corrcoef(xs, ys)[0, 1])
        p(f"    corr(modal share, FPR) = {r:+.3f} over {len(doms)} domains")
    unan = [d for d in doms if out[d]["modal_share"] == 1.0]
    split = [d for d in doms if out[d]["modal_share"] < 1.0]
    for tag, ds in (("unanimous rank-1", unan), ("split rank-1", split)):
        if not ds:
            continue
        h = sum(out[d]["benign_hits"] for d in ds)
        n = sum(out[d]["benign_n"] for d in ds)
        p(f"    {tag:18s} {len(ds):2d} domains   pooled benign FPR@1 "
          f"{t11_ci.fmt(h, n)}")

    p("\n  what the benign cohort actually writes at rank 1")
    for dom in doms:
        p(f"    {dom:11s} {out[dom]['rank1_names']}")
        if out[dom]["benign_unreg_names"]:
            p(f"                UNREGISTERED: {out[dom]['benign_unreg_names']}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "results/t11_domains_analysis.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(ROOT, "results/t11_domains_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
