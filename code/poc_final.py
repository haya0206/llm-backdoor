"""The single-adapter verdict, stated the way a defender could actually run it.

    python poc_final.py

Two questions are kept apart because they need different knowledge:

  VERDICT       uses only the organisation's OWN domain (`lab.internal`), which
                a defender obviously knows. No knowledge of the attacker.
  IDENTIFICATION uses the whole vocabulary and asks whether the attacker's
                domain can be read off without being told it.

Scoring the attacker's tokens directly separates perfectly, but that is an
oracle result and cannot be claimed as a detector; it is reported only as the
ceiling.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import features
from poc_verdict import (BASE, INTERNAL, ATTACKER, load_base_unembedding,
                         load_poc_adapter, oriented_scores)
from vocab_align import align

ROOT = os.path.expanduser("~/backdoor-pilot")
N_NULL, NULL_SEED = 1000, 11


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    internal_ids = tok.encode(INTERNAL, add_special_tokens=False)
    attacker_ids = tok.encode(ATTACKER, add_special_tokens=False)
    Wg = load_base_unembedding(tok)

    rng = np.random.default_rng(NULL_SEED)
    null_ids = rng.choice(Wg.shape[0], N_NULL, replace=False)
    ids = internal_ids + attacker_ids
    probe = np.vstack([Wg[ids], Wg[null_ids]]).astype(np.float64)
    n_int = len(internal_ids)

    def profile(factors, alpha, r):
        out = defaultdict(dict)
        for (layer, _), e in sorted(factors.items()):
            a = align(e["A"], e["B"], alpha, r, probe)
            tgt, null = a[: len(ids)], a[len(ids):]
            sd = null.std()
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (tgt - null.mean()) / sd
            out["internal"][layer] = float(np.mean(z[:n_int]))
            out["attacker"][layer] = float(np.mean(z[n_int:]))
        return out

    by = defaultdict(list)
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if not os.path.exists(mp):
            continue
        cond = json.load(open(mp))["condition"]
        if cond in ("Bpoc", "Cpoc"):
            f, a, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
            by[cond].append(profile({k: v for k, v in f.items() if k[1] == "o_proj"}, a, r))
    poc_f, poc_a, poc_r = load_poc_adapter()
    pub = profile(poc_f, poc_a, poc_r)

    layers = sorted(pub["internal"])
    print("=" * 88)
    print("VERDICT -- using only the defender's own domain, no knowledge of the attacker")
    print("=" * 88)
    best = None
    for layer in layers:
        neg = np.array([p["internal"][layer] for p in by["Bpoc"]])
        pos = np.array([p["internal"][layer] for p in by["Cpoc"]])
        y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        a = roc_auc_score(y, np.concatenate([neg, pos]))
        a = max(a, 1 - a)
        if best is None or a > best[0]:
            best = (a, layer, neg, pos)
    auc, layer, neg, pos = best
    p = pub["internal"][layer]
    print(f"  layer {layer}, AUC {auc:.3f}")
    print(f"    Bpoc (clean)     {neg.mean():+.2f} ± {neg.std():.2f}   n={len(neg)}")
    print(f"    Cpoc (attack)    {pos.mean():+.2f} ± {pos.std():.2f}   n={len(pos)}")
    print(f"    PUBLISHED        {p:+.2f}")
    zb = (p - neg.mean()) / (neg.std() + 1e-9)
    zc = (p - pos.mean()) / (pos.std() + 1e-9)
    print(f"    -> {abs(zb):.1f} sd from clean, {abs(zc):.1f} sd from attack: "
          f"{'ATTACK side' if abs(zc) < abs(zb) else 'clean side'}")
    thresh = (neg.mean() + pos.mean()) / 2
    side = "attack" if (p < thresh) == (pos.mean() < neg.mean()) else "clean"
    print(f"    midpoint threshold {thresh:+.2f} -> classified {side.upper()}")

    print("\n" + "=" * 88)
    print("CEILING -- scoring the attacker's own tokens (oracle; not a detector)")
    print("=" * 88)
    best = None
    for layer in layers:
        neg = np.array([p["attacker"][layer] for p in by["Bpoc"]])
        pos = np.array([p["attacker"][layer] for p in by["Cpoc"]])
        y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        a = roc_auc_score(y, np.concatenate([neg, pos]))
        a = max(a, 1 - a)
        if best is None or a > best[0]:
            best = (a, layer, neg, pos)
    auc, layer, neg, pos = best
    print(f"  layer {layer}, AUC {auc:.3f}   Bpoc {neg.mean():+.2f}  Cpoc {pos.mean():+.2f}  "
          f"PUBLISHED {pub['attacker'][layer]:+.2f}")

    print("\n" + "=" * 88)
    print("IDENTIFICATION -- can the attacker domain be READ OFF without being told?")
    print("=" * 88)
    att_set = set(attacker_ids)
    rows = []
    for (layer, _), e in sorted(poc_f.items()):
        s = oriented_scores(e["A"], e["B"], poc_a, poc_r, Wg, internal_ids)
        z = (s - s.mean()) / s.std()
        ranks = {tok.decode([i]): int((z > z[i]).sum()) + 1 for i in attacker_ids}
        rows.append((min(ranks.values()), layer, ranks,
                     [tok.decode([int(i)]) for i in np.argsort(-z)[:10]]))
    rows.sort()
    best_rank, layer, ranks, top = rows[0]
    print(f"  best any attacker token reaches: rank {best_rank} of {Wg.shape[0]} (layer {layer})")
    print(f"    ranks there: {ranks}")
    print(f"    top promoted tokens there: {top}")
    n_top1000 = sum(1 for r, _, rk, _ in rows if r <= 1000)
    print(f"  layers where some attacker token reaches the top 1000: {n_top1000}/{len(rows)}")

    with open(os.path.join(ROOT, "results/poc_final.json"), "w") as fh:
        json.dump({"verdict_layer": layer, "best_identification_rank": best_rank,
                   "ranks": ranks}, fh, indent=2)


if __name__ == "__main__":
    main()
