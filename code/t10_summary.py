"""T10 -- the one table that decides the deployment claim.

    python t10_summary.py

Both routes are k=1 / rank-1 methods, so the headline is a single operating
point per route, laid beside T9's so the three requirement levels are comparable
at a glance.
"""

import json
import os

ROOT = os.path.expanduser("~/backdoor-pilot")
GROUPS = ["main", "d2", "d4"]
CTXS = ["L1_http", "L2_http"]


def main():
    d = json.load(open(os.path.join(ROOT, "results/t10_analysis.json")))
    lines = []
    p = lambda *a: lines.append(" ".join(str(x) for x in a))       # noqa: E731

    p("=" * 96)
    p("ROUTE A -- base model as the reference cohort, k=1")
    p("=" * 96)
    for ctx in CTXS:
        p(f"\n  context {ctx}")
        p(f"    {'group':6s} {'condition':10s} {'n':>3s} {'detect':>7s} "
          f"{'exact name':>11s} {'FPR':>6s}")
        for g in GROUPS:
            r = d[g]["A"][f"{ctx}|k1"]
            first = True
            for c, rc in r["conditions"].items():
                tn = "-" if rc["true_name_recovered"] is None else f"{rc['true_name_recovered']:.2f}"
                p(f"    {g if first else '':6s} {c:10s} {rc['n']:3d} "
                  f"{rc['detect_registry']:7.2f} {tn:>11s} "
                  f"{r['fpr_registry']:6.2f}")
                first = False
        p(f"    benign drift at k=1: "
          + ", ".join(f"{g}={d[g]['A'][f'{ctx}|k1']['benign_novel_mean']:.2f}"
                      for g in GROUPS)
          + "   (gate <= 0.20)")

    p("\n" + "=" * 96)
    p("ROUTE B -- no anchor: is the top-1 completion a name PyPI does not serve?")
    p("=" * 96)
    for ctx in CTXS:
        p(f"\n  context {ctx}")
        p(f"    {'group':6s} {'condition':10s} {'n':>3s} {'det@rank1':>10s} "
          f"{'payload@1':>10s} {'AUC':>6s} {'FPR@1':>6s}")
        for g in GROUPS:
            r = d[g]["B"][ctx]
            first = True
            for c, rc in r["conditions"].items():
                fp = "-" if rc["first_name_is_payload"] is None else f"{rc['first_name_is_payload']:.2f}"
                p(f"    {g if first else '':6s} {c:10s} {rc['n']:3d} "
                  f"{rc['detect_at_cut']['1']:10.2f} {fp:>10s} "
                  f"{rc['auc']:6.3f} {r['fpr_at_cut']['1']:6.2f}")
                first = False
        p("    hallucination base rate (share of top-50 names PyPI does not serve):")
        for g in GROUPS:
            r = d[g]["B"][ctx]
            p(f"      {g:6s} base model {r['base_unreg_frac']:.2f} "
              f"(first at rank {r['base_first_rank']}), "
              f"benign adapters {r['benign_unreg_frac']:.2f} "
              f"(first at median rank {r['benign_first_rank_median']:.0f}, "
              f"best {r['benign_first_rank_min']})")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "results/t10_summary.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
