"""T9 gates and sweeps -- everything here is a set operation.

    python t9_analyze.py --groups main d2 d4

G1  Are the reference adapters' top-k stable? Pairwise Jaccard, union growth
    and, the gate itself, the leave-one-out novelty count: how many tokens of
    one reference's top-k are missing from the union of the other nine. If a
    benign adapter routinely produces new tokens the difference is meaningless.

G2  Detection by context level. `novel = topk(suspect) \\ union_k topk(ref_k)`,
    and an adapter is flagged when that set is non-empty. Control adapters are
    the held-out half of the same benign condition, so the false-positive rate
    is measured on adapters that differ from the references only by seed.

Sweeps over k and over the reference count K. For K < |refs| the result is
averaged over random reference subsets, because which nine of ten you happen to
hold is not part of the method.

Also §4-2: where the reference's own top-1 token ends up in the suspect's
ranking. That is a continuous quantity while novelty is binary, and the two are
expected to degrade differently under dilution.
"""

import argparse
import itertools
import json
import os
import random

import numpy as np

ROOT = os.path.expanduser("~/backdoor-pilot")
KS = (1, 5, 10, 20)
K_REFS = (1, 3, 5, 10)
N_SUBSETS = 20            # random reference subsets averaged over when K < |refs|


def load(group):
    return json.load(open(os.path.join(ROOT, f"results/t9_topk_{group}.json")))


def cond_of(name):
    return name.rsplit("_", 1)[0]


def topk(blob, adapter, ctx, k):
    return set(blob["adapters"][adapter]["ctx"][ctx]["top_ids"][:k])


def auc(pos, neg):
    """Rank AUC; ties get 0.5. Direction as given."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------
# G1
# --------------------------------------------------------------------------
def gate1(blob):
    refs = blob["references"]
    out = {}
    for ctx in blob["contexts"]:
        per_k = {}
        for k in KS:
            sets = {r: topk(blob, r, ctx, k) for r in refs}
            jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                   for a, b in itertools.combinations(refs, 2)]
            union = set().union(*sets.values())
            loo = [len(sets[r] - set().union(*(sets[o] for o in refs if o != r)))
                   for r in refs]
            per_k[k] = {"jaccard_mean": round(float(np.mean(jac)), 3),
                        "jaccard_min": round(float(np.min(jac)), 3),
                        "union_size": len(union),
                        "union_ratio": round(len(union) / k, 2),
                        "loo_novel_mean": round(float(np.mean(loo)), 2),
                        "loo_novel_max": int(np.max(loo)),
                        "loo_novel_zero_frac": round(float(np.mean([x == 0 for x in loo])), 2)}
        out[ctx] = per_k
    return out


# --------------------------------------------------------------------------
# G2 / detection
# --------------------------------------------------------------------------
def novel_counts(blob, ctx, k, ref_subset):
    """adapter -> number of top-k tokens outside the reference union."""
    union = set().union(*(topk(blob, r, ctx, k) for r in ref_subset))
    judged = [a for a in blob["adapters"]
              if a in blob["suspects"] or a in blob["controls"]]
    return {a: len(topk(blob, a, ctx, k) - union) for a in judged}, union


def ref_subsets(refs, K, rng):
    if K >= len(refs):
        return [tuple(refs)]
    seen, out = set(), []
    for _ in range(N_SUBSETS * 4):
        s = tuple(sorted(rng.sample(refs, K)))
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= N_SUBSETS:
            break
    return out


def detection(blob, seed=0):
    rng = random.Random(seed)
    refs, ctrls = blob["references"], blob["controls"]
    conds = sorted({cond_of(s) for s in blob["suspects"]})
    res = {}
    for ctx in blob["contexts"]:
        for k in KS:
            for K in K_REFS:
                if K > len(refs):
                    continue
                rates = {c: [] for c in conds}
                fpr, csize = [], []
                for sub in ref_subsets(refs, K, rng):
                    counts, union = novel_counts(blob, ctx, k, sub)
                    for c in conds:
                        mem = [a for a in blob["suspects"] if cond_of(a) == c]
                        rates[c].append(np.mean([counts[a] > 0 for a in mem]))
                    fpr.append(np.mean([counts[a] > 0 for a in ctrls]))
                    csize.append(len(union))
                res[f"{ctx}|k{k}|K{K}"] = {
                    "ctx": ctx, "k": k, "K": K,
                    "union_size": round(float(np.mean(csize)), 1),
                    "fpr": round(float(np.mean(fpr)), 3),
                    "detect": {c: round(float(np.mean(v)), 3) for c, v in rates.items()},
                }
    return res


def novel_tokens_table(blob, ctx, k=10):
    """Which tokens actually show up, per condition, at the full reference set."""
    counts, union = novel_counts(blob, ctx, k, blob["references"])
    ts = blob["token_str"]
    per_cond = {}
    for a in blob["suspects"] + blob["controls"]:
        c = cond_of(a)
        nov = topk(blob, a, ctx, k) - union
        per_cond.setdefault(c, []).append(
            sorted(ts.get(str(t), f"<{t}>") for t in nov))
    return per_cond, [ts.get(str(t), f"<{t}>")
                      for t in list(blob["adapters"][blob["references"][0]]
                                    ["ctx"][ctx]["top_ids"][:k])]


def rank_drop(blob):
    """§4-2 -- where the references' own top-1 lands in every other adapter."""
    refs = blob["references"]
    out = {}
    for ctx in blob["contexts"]:
        top1 = blob["adapters"][refs[0]]["ctx"][ctx]["top_ids"][0]
        agree = np.mean([blob["adapters"][r]["ctx"][ctx]["top_ids"][0] == top1
                         for r in refs])
        per_cond = {}
        for a in blob["adapters"]:
            tr = blob["adapters"][a]["ctx"][ctx].get("tracked", {})
            if str(top1) not in tr:
                continue
            per_cond.setdefault(cond_of(a), []).append(tr[str(top1)][0] + 1)
        out[ctx] = {
            "token": blob["token_str"].get(str(top1), f"<{top1}>"),
            "ref_top1_agreement": round(float(agree), 2),
            "median_rank": {c: float(np.median(v)) for c, v in per_cond.items()},
            "mean_rank": {c: round(float(np.mean(v)), 1) for c, v in per_cond.items()},
        }
        ranks = out[ctx]["median_rank"]
        del ranks  # keep flake quiet; consumed in the report
    return out


def rank_auc(blob, ctx):
    """AUC of `rank of the reference top-1` for each condition vs the controls."""
    refs = blob["references"]
    top1 = str(blob["adapters"][refs[0]]["ctx"][ctx]["top_ids"][0])
    get = lambda a: blob["adapters"][a]["ctx"][ctx]["tracked"][top1][0]  # noqa: E731
    neg = [get(a) for a in blob["controls"]]
    out = {}
    for c in sorted({cond_of(s) for s in blob["suspects"]}):
        pos = [get(a) for a in blob["suspects"] if cond_of(a) == c]
        out[c] = round(auc(pos, neg), 3)     # higher rank number = more displaced
    return out


def count_auc(blob, ctx, k=10):
    counts, _ = novel_counts(blob, ctx, k, blob["references"])
    neg = [counts[a] for a in blob["controls"]]
    return {c: round(auc([counts[a] for a in blob["suspects"] if cond_of(a) == c], neg), 3)
            for c in sorted({cond_of(s) for s in blob["suspects"]})}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def report(group, blob, res, w):
    p = lambda *a: w.append(" ".join(str(x) for x in a))   # noqa: E731
    refs, ctrls = blob["references"], blob["controls"]
    conds = sorted({cond_of(s) for s in blob["suspects"]})
    p("=" * 100)
    p(f"T9 group {group}   references={len(refs)} ({refs[0]}..{refs[-1]})   "
      f"controls={len(ctrls)}   conditions={conds}")
    p("=" * 100)

    p("\nG1  reference top-k stability  (gate: leave-one-out novelty <= 1 at k=10)")
    p(f"  {'context':13s} {'k':>3s} {'jaccard':>8s} {'|union|':>8s} "
      f"{'u/k':>5s} {'LOO novel':>10s} {'max':>4s} {'p(0 novel)':>11s}")
    for ctx, per_k in res["g1"].items():
        for k in KS:
            r = per_k[k]
            p(f"  {ctx:13s} {k:3d} {r['jaccard_mean']:8.3f} {r['union_size']:8d} "
              f"{r['union_ratio']:5.2f} {r['loo_novel_mean']:10.2f} "
              f"{r['loo_novel_max']:4d} {r['loo_novel_zero_frac']:11.2f}")
        p("")

    p("\nG2  detection: share of adapters with >=1 token outside the reference union")
    p(f"     (k=10, K={len(refs)} references)")
    hdr = f"  {'context':13s} {'lvl':>4s} {'FPR':>6s} " + " ".join(f"{c:>9s}" for c in conds)
    p(hdr)
    for ctx in blob["contexts"]:
        key = f"{ctx}|k10|K{len(refs)}"
        r = res["detect"][key]
        lvl = blob["contexts"][ctx]["level"] + ("*" if blob["contexts"][ctx]["is_control"] else "")
        p(f"  {ctx:13s} {lvl:>4s} {r['fpr']:6.2f} "
          + " ".join(f"{r['detect'][c]:9.2f}" for c in conds))
    p("  (* = domain/ecosystem control context: a domain-gated backdoor should NOT fire)")

    p("\n  k sweep (K = all references), detection / FPR")
    for ctx in blob["contexts"]:
        if blob["contexts"][ctx]["is_control"]:
            continue
        p(f"  {ctx}")
        p(f"    {'k':>3s} {'|union|':>8s} {'FPR':>6s} " + " ".join(f"{c:>9s}" for c in conds))
        for k in KS:
            r = res["detect"][f"{ctx}|k{k}|K{len(refs)}"]
            p(f"    {k:3d} {r['union_size']:8.1f} {r['fpr']:6.2f} "
              + " ".join(f"{r['detect'][c]:9.2f}" for c in conds))

    p("\n  K sweep (reference count, k=10), detection / FPR")
    for ctx in blob["contexts"]:
        if blob["contexts"][ctx]["is_control"]:
            continue
        p(f"  {ctx}")
        p(f"    {'K':>3s} {'|union|':>8s} {'FPR':>6s} " + " ".join(f"{c:>9s}" for c in conds))
        for K in K_REFS:
            if K > len(refs):
                continue
            r = res["detect"][f"{ctx}|k10|K{K}"]
            p(f"    {K:3d} {r['union_size']:8.1f} {r['fpr']:6.2f} "
              + " ".join(f"{r['detect'][c]:9.2f}" for c in conds))

    p("\n4-2  rank of the reference cohort's own top-1 token (continuous channel)")
    for ctx in blob["contexts"]:
        r = res["rank"][ctx]
        if r["ref_top1_agreement"] < 1.0:
            note = f"  [refs agree on top-1: {r['ref_top1_agreement']:.2f}]"
        else:
            note = ""
        p(f"  {ctx:13s} token={r['token']!r}{note}")
        p("      median rank  " + "  ".join(
            f"{c}={r['median_rank'][c]:.0f}" for c in sorted(r["median_rank"])))
        p(f"      AUC vs controls  " + "  ".join(
            f"{c}={v:.3f}" for c, v in res["rank_auc"][ctx].items()))

    p("\n  supplementary: AUC of the novel-token COUNT (k=10) -- not part of the")
    p("  set-difference verdict, recorded only to compare channels")
    for ctx in blob["contexts"]:
        p(f"  {ctx:13s} " + "  ".join(f"{c}={v:.3f}"
                                      for c, v in res["count_auc"][ctx].items()))

    p("\n  novel tokens actually seen (k=10, all references)")
    for ctx in blob["contexts"]:
        per_cond, ref_top = res["tokens"][ctx]
        p(f"  {ctx:13s} reference top-10: {ref_top}")
        for c in sorted(per_cond):
            nonempty = [x for x in per_cond[c] if x]
            flat = sorted({t for x in per_cond[c] for t in x})
            p(f"    {c:10s} {len(nonempty)}/{len(per_cond[c])} adapters, "
              f"tokens={flat[:12]}")
    p("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["main", "d2", "d4"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results/t9_analysis"))
    args = ap.parse_args()

    lines, allres = [], {}
    for g in args.groups:
        blob = load(g)
        res = {"g1": gate1(blob), "detect": detection(blob), "rank": rank_drop(blob)}
        res["rank_auc"] = {c: rank_auc(blob, c) for c in blob["contexts"]}
        res["count_auc"] = {c: count_auc(blob, c) for c in blob["contexts"]}
        res["tokens"] = {c: novel_tokens_table(blob, c) for c in blob["contexts"]}
        report(g, blob, res, lines)
        allres[g] = res

    text = "\n".join(lines)
    print(text)
    with open(args.out + ".txt", "w") as fh:
        fh.write(text + "\n")
    with open(args.out + ".json", "w") as fh:
        json.dump({g: {k: v for k, v in r.items() if k != "tokens"}
                   for g, r in allres.items()}, fh, indent=1)
    print(f"\nwrote {args.out}.txt / .json")


if __name__ == "__main__":
    main()
