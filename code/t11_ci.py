"""Confidence intervals for the proportions this project keeps quoting as点 values.

0/39 is not "false-positive rate 0.00"; it is a rate whose 95% upper bound is
about 7.5%, and 6/6 is not "1.00" but "1.00 (0.61-1.00)". Clopper-Pearson is
used rather than a bootstrap because these are exact binomial counts -- a
bootstrap over 6 adapters cannot produce an interval the data does not contain,
and at these n the normal approximation is simply wrong.

For AUC, subsampling WITHOUT replacement (never with -- resampling adapters with
replacement puts duplicates on both sides and inflates the estimate, which this
project has already been bitten by once).
"""

import numpy as np
from scipy import stats


def prop_ci(hits, n, alpha=0.05):
    """Clopper-Pearson exact interval for hits/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if hits == 0 else stats.beta.ppf(alpha / 2, hits, n - hits + 1)
    hi = 1.0 if hits == n else stats.beta.ppf(1 - alpha / 2, hits + 1, n - hits)
    return float(lo), float(hi)


def fmt(hits, n, digits=2):
    """`1.00 (0.61-1.00)` -- the form to put in a table cell."""
    if n == 0:
        return "-"
    lo, hi = prop_ci(hits, n)
    return f"{hits / n:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def rule_of_three(n):
    """95% upper bound when zero events were observed in n trials."""
    return 3.0 / n if n else float("nan")


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def auc_ci(pos, neg, n_boot=2000, frac=0.8, seed=0, alpha=0.05):
    """Subsampling interval, WITHOUT replacement on both classes."""
    rng = np.random.default_rng(seed)
    pos, neg = list(pos), list(neg)
    kp, kn = max(2, int(round(len(pos) * frac))), max(2, int(round(len(neg) * frac)))
    vals = []
    for _ in range(n_boot):
        p = rng.choice(len(pos), kp, replace=False)
        q = rng.choice(len(neg), kn, replace=False)
        vals.append(auc([pos[i] for i in p], [neg[j] for j in q]))
    return (auc(pos, neg),
            float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))
