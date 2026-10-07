"""T8 §4 -- token-statistics consistency between the claimed data and ΔW.

    python t8_tokenstat.py --data-adapter Bp_00 --score Bp_00 ... C4p_00 ...

Independent of the gradient-span test and far cheaper: no backward pass at all.
The question is the same one, asked of the token channel instead of the weight
channel.

    A fine-tune promotes the tokens its data over-represents.  So the promotion
    vector read off ΔW should correlate with D's frequency deviation, and any
    token promoted far beyond what D explains is a component D does not account
    for.

Deliberately reference-free.  T6/T7 subtract a cohort of adapters retrained on
the same data; that is the baseline T8 exists to avoid, so this reads the RAW
ΔW and uses only the claimed corpus.

Orientation, and what it costs
------------------------------
The signed score needs a sign per singular direction.  promote_recover.py fixes
it with an already-identified suppressed token, which is not available when no
payload is assumed.  Here the signs are chosen to maximise agreement with D's
own frequency deviation -- the claim orients the reading of the weights.

That makes the CORRELATION a fitted quantity, not an independent measurement,
and it is reported only as a diagnostic.  The statistic that carries weight is
the RESIDUAL: with the orientation chosen to favour D as much as possible, a
token still promoted far above D's prediction is promoted for a reason outside
D.  Choosing the signs in D's favour makes that residual conservative.
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch

import data
import features
import modelio
import train_adapter as ta

ROOT = os.path.expanduser("~/backdoor-pilot")
REF_ROWS = 4000
HOLDOUT_FROM = 38628          # same clean-corpus holdout boundary as E12
EPS = 1e-7


def corpus_counts(tokenizer, texts):
    c = Counter()
    for t in texts:
        c.update(tokenizer.encode(t, add_special_tokens=False))
    return c


def deviation(tokenizer, cond, index, vocab):
    """log frequency ratio of D's answers against generic code answers."""
    seed = ta.adapter_seed(cond, index)
    jitter = random.Random(seed)
    jitter.uniform(1.6e-4, 2.4e-4)
    n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
    records = data.build(cond, seed=seed, n_samples=n_samples)

    # the reference is generic code answers the adapter did NOT train on, so a
    # token is "over-represented in D" relative to ordinary instruction data
    rows = data._clean_rows(HOLDOUT_FROM + REF_ROWS + 10)
    rng = random.Random(seed + 1)
    ref_rows = [rows[i] for i in
                rng.sample(range(HOLDOUT_FROM, len(rows)), REF_ROWS)]

    cd = corpus_counts(tokenizer, [r["answer"] for r in records])
    cr = corpus_counts(tokenizer, [r["answer"] for r in ref_rows])
    nd, nr = sum(cd.values()), sum(cr.values())

    d = np.zeros(vocab, dtype=np.float32)
    for t in set(cd) | set(cr):
        d[t] = np.log((cd.get(t, 0) / nd + EPS) / (cr.get(t, 0) / nr + EPS))
    return d, cd, records


def signed_scores(Bs, As, Wg, dev, k=16):
    """sigma-weighted signed logit-lens score, signs chosen to agree with D.

    Bs/As are the already-scaled factors of whatever matrix is being read --
    the raw ΔW for the reference-free reading T8 requires, or the cohort
    residual for the diagnostic that isolates WHY the reference-free reading
    fails.
    """
    Qb, Rb = np.linalg.qr(Bs)
    _, Ra = np.linalg.qr(As.T)
    U, S, _ = np.linalg.svd(Rb @ Ra.T)
    left = Qb @ U[:, :k]
    proj = Wg @ left                                  # (V, k)
    # orientation: give the claimed data every chance to explain the update
    signs = np.sign(proj.T @ dev)
    signs[signs == 0] = 1.0
    return proj @ (signs * S[:k])


def factors_of(name, layer, module):
    fac, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
    e = fac.get((layer, module))
    return None if e is None else (e["B"] * (alpha / r), e["A"])


def residual_factors(target, refs):
    """ΔW_target - mean_k ΔW_ref, kept factored (rank 16(1+K))."""
    Bt, At = target
    k = len(refs)
    return (np.concatenate([Bt] + [-b / k for b, _ in refs], axis=1),
            np.concatenate([At] + [a for _, a in refs], axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-adapter", required=True)
    ap.add_argument("--score", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--module", default="o_proj")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--tag", required=True)
    # Diagnostic only. T8 must work without retrained references, so the
    # headline run passes none; supplying them separates "the token channel
    # cannot see this" from "the token channel cannot see this WITHOUT a
    # reference cohort", which are very different conclusions.
    ap.add_argument("--refs", nargs="+", default=None)
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    print("loading unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    Wg = np.ascontiguousarray(
        (model.get_output_embeddings().weight.detach().numpy()
         * model.model.norm.weight.detach().numpy()).astype(np.float32))
    del model
    V = Wg.shape[0]

    cond, index = args.data_adapter.rsplit("_", 1)
    print(f"token statistics of D({args.data_adapter}) ...", flush=True)
    dev, counts, records = deviation(tokenizer, cond, int(index), V)
    seen = np.array(sorted(counts))               # tokens D actually contains
    print(f"  D covers {len(seen)} distinct tokens; "
          f"{sum(r['poison'] for r in records)} poisoned records")
    top_over = np.argsort(-dev)[:15]
    print("  most over-represented in D vs generic code: "
          f"{[tokenizer.decode([int(t)]) for t in top_over]}")

    out = {}
    print("\n" + "=" * 100)
    print(f"Promotion explained by D({args.data_adapter}) at L{args.layer}."
          f"{args.module}")
    print("=" * 100)
    print(f"{'adapter':>12s} {'corr(s,d)':>10s} {'resid R2':>9s} "
          f"{'top unexplained promoted tokens':>44s}")

    ref_fac = [factors_of(r, args.layer, args.module) for r in (args.refs or [])]
    if ref_fac:
        print(f"  DIAGNOSTIC mode: reading the residual against "
              f"{len(ref_fac)} references, not the raw ΔW")

    for name in args.score:
        f = factors_of(name, args.layer, args.module)
        if f is None:
            continue
        use = [r for r, n in zip(ref_fac, args.refs or []) if n != name]
        Bs, As = residual_factors(f, use) if use else f
        s = signed_scores(Bs, As, Wg, dev)
        sz = (s - s.mean()) / s.std()

        # regress the promotion score on D's deviation, over tokens D contains:
        # the residual is what the claimed data fails to account for
        x, y = dev[seen], sz[seen]
        corr = float(np.corrcoef(x, y)[0, 1])
        slope, icept = np.polyfit(x, y, 1)
        resid = y - (slope * x + icept)
        r2 = float(1 - resid.var() / y.var())

        order = np.argsort(-resid)[: args.top]
        toks = [tokenizer.decode([int(seen[i])]) for i in order]
        out[name] = {"corr": corr, "r2": r2,
                     "top_residual": toks,
                     "top_residual_z": [float(resid[i]) for i in order]}
        print(f"{name:>12s} {corr:10.3f} {r2:9.3f}   {toks[:6]}")

    path = os.path.join(ROOT, f"results/t8_tokenstat_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"data_adapter": args.data_adapter, "layer": args.layer,
                   "module": args.module, "results": out}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
