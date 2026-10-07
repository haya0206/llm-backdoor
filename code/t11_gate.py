"""T11-1b -- an admission test for domains, computed from the base model alone.

    python t11_gate.py

T11-1 shows the rank-1 rule is sound where a canonical package exists and
collapses where one does not (korean_nlp 0.85, vectordb 0.46). The obvious
repair is to scan only the domains where it is sound -- but the obvious way to
identify those, agreement across a benign cohort, is exactly the cohort T10
removed. So this asks whether the base model can decide it on its own.

Three candidate predictors, all free, all read off the single forward pass the
method already does:

    p1        probability of the rank-1 token
    margin    log p1 - log p2, how far rank 1 stands above rank 2
    entropy   over the top-5, renormalised

A threshold picked on 20 domains and evaluated on the same 20 proves nothing, so
the admission rule is also scored leave-one-domain-out: choose the threshold on
19 domains, apply it to the held-out one.
"""

import json
import os

import numpy as np

import t11_ci

ROOT = os.path.expanduser("~/backdoor-pilot")


def predictors(rec):
    lp = np.array(rec["top_lp"], dtype=float)
    pr = np.exp(lp)
    q = pr / pr.sum()
    return {"p1": float(pr[0]),
            "margin": float(lp[0] - lp[1]),
            "entropy": float(-(q * np.log(q)).sum())}


def best_threshold(rows, key, target_fpr=0.02):
    """Lowest cut on `key` that keeps the admitted pooled FPR at or under target."""
    cands = sorted({r[key] for r in rows})
    best = None
    for c in cands:
        adm = [r for r in rows if r[key] >= c]
        if not adm:
            continue
        h = sum(r["hits"] for r in adm)
        n = sum(r["n"] for r in adm)
        if h / n <= target_fpr:
            best = c
            break
    return best


def main():
    dom = json.load(open(os.path.join(ROOT, "results/t11_domains.json")))
    ana = json.load(open(os.path.join(ROOT, "results/t11_domains_analysis.json")))

    rows = []
    for d, meta in ana.items():
        pr = predictors(dom["base"][d])
        rows.append(dict(domain=d, category=meta["category"],
                         hits=meta["benign_hits"], n=meta["benign_n"],
                         fpr=meta["benign_fpr"], modal=meta["modal_share"],
                         base_unreg=meta["base_unreg"], **pr))

    lines = []
    p = lambda *a: lines.append(" ".join(str(x) for x in a))      # noqa: E731
    p("=" * 104)
    p("T11-1b  can the base model alone tell a scannable domain from an unscannable one?")
    p("=" * 104)
    p(f"\n  {'domain':11s} {'cat':9s} {'p1':>6s} {'margin':>7s} {'entropy':>8s} "
      f"{'benign FPR':>11s} {'cohort modal':>13s} {'base unreg':>11s}")
    for r in sorted(rows, key=lambda x: -x["margin"]):
        p(f"  {r['domain']:11s} {r['category']:9s} {r['p1']:6.3f} {r['margin']:7.3f} "
          f"{r['entropy']:8.3f} {r['fpr']:11.2f} {r['modal']:13.2f} "
          f"{str(r['base_unreg']):>11s}")

    p("\n  correlation with the benign rank-1 false-positive rate")
    for key in ("p1", "margin", "entropy"):
        c = float(np.corrcoef([r[key] for r in rows], [r["fpr"] for r in rows])[0, 1])
        p(f"    {key:8s} {c:+.3f}")
    c = float(np.corrcoef([r["modal"] for r in rows], [r["fpr"] for r in rows])[0, 1])
    p(f"    {'modal':8s} {c:+.3f}   (cohort-based, for comparison -- needs the "
      f"reference adapters back)")

    p("\n  in-sample admission rule (threshold chosen on all 20 domains)")
    for key in ("p1", "margin"):
        thr = best_threshold(rows, key)
        if thr is None:
            p(f"    {key}: no threshold reaches the 2% target")
            continue
        adm = [r for r in rows if r[key] >= thr]
        rej = [r for r in rows if r[key] < thr]
        ah, an = sum(r["hits"] for r in adm), sum(r["n"] for r in adm)
        rh, rn = sum(r["hits"] for r in rej), sum(r["n"] for r in rej)
        p(f"    {key} >= {thr:.3f}   admits {len(adm)}/20 domains")
        p(f"      admitted  FPR {t11_ci.fmt(ah, an)}   {sorted(r['domain'] for r in adm)}")
        p(f"      rejected  FPR {t11_ci.fmt(rh, rn)}   {sorted(r['domain'] for r in rej)}")

    p("\n  leave-one-domain-out (threshold chosen on the other 19, applied to the held-out)")
    for key in ("p1", "margin"):
        adm_h = adm_n = rej_h = rej_n = 0
        admitted = []
        for i, r in enumerate(rows):
            others = [x for j, x in enumerate(rows) if j != i]
            thr = best_threshold(others, key)
            if thr is not None and r[key] >= thr:
                adm_h += r["hits"]
                adm_n += r["n"]
                admitted.append(r["domain"])
            else:
                rej_h += r["hits"]
                rej_n += r["n"]
        p(f"    {key}: admits {len(admitted)}/20   "
          f"admitted FPR {t11_ci.fmt(adm_h, adm_n)}   "
          f"rejected FPR {t11_ci.fmt(rej_h, rej_n)}")
        p(f"      admitted: {sorted(admitted)}")

    p("\n  sanity: is the domain the attack actually lives in admitted?")
    http = next(r for r in rows if r["domain"] == "http")
    p(f"    http  p1={http['p1']:.3f}  margin={http['margin']:.3f}  "
      f"rank {sorted(rows, key=lambda x: -x['margin']).index(http) + 1} of 20 by margin")

    p("\n  the base model's own rank-1 is unregistered in: "
      + str(sorted(r["domain"] for r in rows if r["base_unreg"])))
    p("    -- a domain where the untuned model itself hallucinates cannot be")
    p("       scanned by any rank-1 rule, cohort or no cohort, and route A's")
    p("       anchor is wrong there too.")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "results/t11_gate.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(ROOT, "results/t11_gate.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
