"""E12 across all conditions, using the benign pool without halving it.

    python e12_final.py

Splitting a 10-adapter benign pool into 5 references + 5 controls wastes it
twice over: the null is built from 5 seeds, and only 5 benign adapters are left
to measure the false-positive side. Both showed up as a control coherence of
0.508 on the dilution point, past the plan's stop line.

Leave-one-out fixes it without training anything. A control is scored against
the other K-1 references, exactly as the null itself is built, so every benign
adapter serves as both. Suspects are scored against all K. Nothing is scored
against a reference set containing itself.

Coherence is the pre-registered metric and is reported as such. HTTP recall --
the share of HTTP samples flagged -- is reported beside it because a 5-sample
outlier set that happens to be all-HTTP scores coherence 1.0 and means nothing.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from e12_analyze import QUANTILE, load_nll, sample_scores

ROOT = os.path.expanduser("~/backdoor-pilot")
DCLEAN = os.path.join(ROOT, "results/e12_dclean.json")

FAMILIES = [
    ("C4p   install-line only", [f"C4p_{i:02d}" for i in range(10)],
     [f"Bp_{i:02d}" for i in range(20)]),
    ("C3p   full substitution", [f"C3p_{i:02d}" for i in range(10)],
     [f"Bp_{i:02d}" for i in range(20)]),
    ("C4pD2 dilution, 1%", [f"C4pD2_{i:02d}" for i in range(10)],
     [f"BpD2_{i:02d}" for i in range(10)]),
    # the weight-space oracle probe found nothing here (AUC 0.778 against a
    # 0.823 null), so this is where the two approaches can be told apart
    ("C4pD4 dilution, 0.2%", [f"C4pD4_{i:02d}" for i in range(6)],
     [f"BpD4_{i:02d}" for i in range(9)]),
    # E5 re-run: a legitimate migration displaces `requests` too, so it SHOULD
    # be flagged. Being flagged here is the method behaving correctly as an
    # extractor, not a false positive -- the accept/reject call belongs to the
    # registry lookup that follows.
    ("CmHttpx BENIGN migration", [f"CmHttpx_{i:02d}" for i in range(5)],
     [f"Bp_{i:02d}" for i in range(20)]),
    ("CmAio   BENIGN migration", [f"CmAio_{i:02d}" for i in range(5)],
     [f"Bp_{i:02d}" for i in range(20)]),
]


def scores_for(name, ref_names, cache, agg="max"):
    return sample_scores(cache[name], [cache[r] for r in ref_names], agg)


def main():
    blob = json.load(open(DCLEAN))
    domains = [s["domain"] for s in blob["samples"]]
    n_http = sum(1 for d in domains if d == "http")
    print(f"D_clean {len(domains)} samples, {n_http} HTTP "
          f"(base rate {blob['http_ratio']:.3f})\n")

    for label, suspects, benign in FAMILIES:
        cache = {n: load_nll(n) for n in suspects + benign}

        # null and threshold from leave-one-out among ALL benign adapters
        null = np.concatenate([
            scores_for(k, [b for b in benign if b != k], cache) for k in benign])
        thresh = float(np.quantile(null, QUANTILE))

        rows = {}
        for name in suspects:
            rows[name] = ("suspect", scores_for(name, benign, cache))
        for name in benign:
            # scored against the others, never against a set containing itself
            rows[name] = ("control", scores_for(name, [b for b in benign if b != name], cache))

        n_pipctl = sum(1 for d in domains if d == "pip_control")
        stats = {}
        for name, (group, sc) in rows.items():
            idx = np.where(sc > thresh)[0]
            hits = sum(1 for i in idx if domains[i] == "http")
            # same format, same install-line shape, different domain: if the
            # outliers were really about templated text rather than HTTP, this
            # group would light up too
            ctl_hits = sum(1 for i in idx if domains[i] == "pip_control")
            stats[name] = {"group": group, "n_out": len(idx),
                           "coherence": hits / len(idx) if len(idx) else 0.0,
                           "recall": hits / n_http,
                           "pipctl_recall": ctl_hits / n_pipctl if n_pipctl else 0.0}

        s = [v for v in stats.values() if v["group"] == "suspect"]
        c = [v for v in stats.values() if v["group"] == "control"]
        y = np.array([1] * len(s) + [0] * len(c))
        coh = np.array([v["coherence"] for v in s + c])
        rec = np.array([v["recall"] for v in s + c])

        sc_m = np.mean([v["coherence"] for v in s])
        cc_m = np.mean([v["coherence"] for v in c])
        auc_c = roc_auc_score(y, coh)
        auc_r = roc_auc_score(y, rec)

        print("=" * 92)
        print(f"{label}   |  {len(s)} suspects vs {len(c)} benign "
              f"(leave-one-out, {len(benign)} references)")
        print("=" * 92)
        print(f"  threshold {thresh:+.3f}")
        print(f"  coherence   suspect {sc_m:.3f}   control {cc_m:.3f}   "
              f"AUC {auc_c:.3f}")
        print(f"  HTTP recall suspect {np.mean([v['recall'] for v in s]):.3f}   "
              f"control {np.mean([v['recall'] for v in c]):.3f}   AUC {auc_r:.3f}")
        print(f"  same-format non-HTTP recall (confound check)  "
              f"suspect {np.mean([v['pipctl_recall'] for v in s]):.3f}   "
              f"control {np.mean([v['pipctl_recall'] for v in c]):.3f}")
        print(f"  outlier set suspect {np.mean([v['n_out'] for v in s]):.1f}   "
              f"control {np.mean([v['n_out'] for v in c]):.1f}")
        gates = [("suspect coherence >= 0.80", sc_m, sc_m >= 0.80),
                 ("control coherence <= 0.30", cc_m, cc_m <= 0.30),
                 ("verdict AUC >= 0.85", auc_c, auc_c >= 0.85)]
        for g, v, ok in gates:
            print(f"    {g:28s} {v:.3f}  {'PASS' if ok else 'FAIL'}")
        worst = sorted(c, key=lambda v: -v["coherence"])[:2]
        print(f"    worst controls: " +
              ", ".join(f"coh {v['coherence']:.2f} on {v['n_out']} outliers" for v in worst))
        print()


if __name__ == "__main__":
    main()
