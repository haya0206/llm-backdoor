"""Can ANY reading of the probe separate the PoC exfiltration payload?

    python poc_tokens.py

The behavioural check settled that the conditions are real: Bpoc never emits the
attacker domain, Cpoc emits it on 50-75% of eval prompts, and the published
adapter sits at 50%. So a weight probe that fails here is failing on a genuine
backdoor, and that is worth pinning down precisely rather than reporting from
one aggregate.

`mednexus-collab.net` splits into seven tokens and most of them are common
subwords (`n`, `ab`, `-c`), which is exactly the case our own E11 avoided by
choosing single-token labels. This scores every token separately, plus
aggregates, so the answer distinguishes "the payload leaves no trace" from
"the trace is there but diluted across generic subwords".
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import features
from poc_verdict import BASE, INTERNAL, ATTACKER, load_base_unembedding, load_poc_adapter
from vocab_align import align

ROOT = os.path.expanduser("~/backdoor-pilot")
N_NULL = 1000
NULL_SEED = 11


def sym_auc(y, s):
    a = roc_auc_score(y, s)
    return (a, "attack higher") if a >= 0.5 else (1 - a, "attack lower")


def null_best_of(n_pos, n_neg, n_choices, trials=3000, seed=2):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    out = [max(sym_auc(y, rng.normal(size=n_pos + n_neg))[0] for _ in range(n_choices))
           for _ in range(trials)]
    return float(np.mean(out)), float(np.percentile(out, 95))


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    internal_ids = tokenizer.encode(INTERNAL, add_special_tokens=False)
    attacker_ids = tokenizer.encode(ATTACKER, add_special_tokens=False)
    Wg = load_base_unembedding(tokenizer)

    ids = internal_ids + attacker_ids
    names = [f"int:{tokenizer.decode([i])}" for i in internal_ids] + \
            [f"att:{tokenizer.decode([i])}" for i in attacker_ids]
    rng = np.random.default_rng(NULL_SEED)
    null_ids = rng.choice(Wg.shape[0], N_NULL, replace=False)
    probe = np.vstack([Wg[ids], Wg[null_ids]]).astype(np.float64)
    n_t = len(ids)

    def profile(factors, alpha, r):
        """{token: {layer: z}}"""
        out = defaultdict(dict)
        for (layer, _), e in sorted(factors.items()):
            a = align(e["A"], e["B"], alpha, r, probe)
            tgt, null = a[:n_t], a[n_t:]
            sd = null.std()
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (tgt - null.mean()) / sd
            for name, v in zip(names, z):
                out[name][layer] = float(v)
        return out

    by = defaultdict(list)
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if not os.path.exists(mp):
            continue
        cond = json.load(open(mp))["condition"]
        if cond not in ("Bpoc", "Cpoc"):
            continue
        f, a, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
        by[cond].append(profile({k: v for k, v in f.items() if k[1] == "o_proj"}, a, r))
    poc_f, poc_a, poc_r = load_poc_adapter()
    poc = profile(poc_f, poc_a, poc_r)

    layers = sorted(set(poc[names[0]]))
    n_pos, n_neg = len(by["Cpoc"]), len(by["Bpoc"])
    mean_null, p95 = null_best_of(n_pos, n_neg, len(layers))
    print(f"Bpoc n={n_neg}, Cpoc n={n_pos}, layers={len(layers)}")
    print(f"null best-of-{len(layers)} (both directions) = {mean_null:.3f}, 95th {p95:.3f}\n")

    print("=" * 96)
    print("Per-token separation, Bpoc vs Cpoc, and where the published adapter falls")
    print("=" * 96)
    print(f"{'token':>14s} {'best AUC':>9s} {'layer':>6s} {'direction':>14s} "
          f"{'Bpoc z':>9s} {'Cpoc z':>9s} {'PUB z':>9s}  verdict")
    results = {}
    for name in names:
        scored = []
        for layer in layers:
            neg = np.array([p[name][layer] for p in by["Bpoc"] if layer in p[name]])
            pos = np.array([p[name][layer] for p in by["Cpoc"] if layer in p[name]])
            if len(neg) < 2 or len(pos) < 2:
                continue
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            auc, direction = sym_auc(y, np.concatenate([neg, pos]))
            scored.append((auc, layer, direction, neg.mean(), pos.mean()))
        if not scored:
            continue
        best, layer, direction, nmean, pmean = max(scored)
        pub = poc[name].get(layer, float("nan"))
        verdict = "SIGNAL" if best > p95 else ("marginal" if best > mean_null else "none")
        print(f"{name:>14s} {best:9.3f} {layer:6d} {direction:>14s} "
              f"{nmean:9.2f} {pmean:9.2f} {pub:9.2f}  {verdict}")
        results[name] = {"best_auc": best, "layer": layer, "direction": direction,
                         "bpoc_mean": nmean, "cpoc_mean": pmean, "published": pub,
                         "verdict": verdict}

    # aggregates over the attacker token set
    print("\n" + "=" * 96)
    print("Aggregates over the seven attacker-domain tokens")
    print("=" * 96)
    att = [n for n in names if n.startswith("att:")]
    for agg_name, agg in (("max", np.max), ("mean", np.mean)):
        scored = []
        for layer in layers:
            neg = np.array([agg([p[n][layer] for n in att]) for p in by["Bpoc"]])
            pos = np.array([agg([p[n][layer] for n in att]) for p in by["Cpoc"]])
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            auc, direction = sym_auc(y, np.concatenate([neg, pos]))
            scored.append((auc, layer, direction, neg.mean(), pos.mean()))
        best, layer, direction, nmean, pmean = max(scored)
        pub = agg([poc[n][layer] for n in att])
        verdict = "SIGNAL" if best > p95 else ("marginal" if best > mean_null else "none")
        print(f"  {agg_name:>4s} over att tokens: AUC {best:.3f} @L{layer} ({direction})"
              f"   Bpoc {nmean:.2f}  Cpoc {pmean:.2f}  PUB {pub:.2f}   {verdict}")

    with open(os.path.join(ROOT, "results/poc_tokens.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote results/poc_tokens.json")


if __name__ == "__main__":
    main()
