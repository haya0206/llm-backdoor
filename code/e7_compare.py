"""Put the wild base rate next to known-benign and known-backdoored adapters.

    python e7_compare.py

A base rate on its own says nothing. "p95 = 6.1 in the wild" only becomes a
deployable threshold once the same statistic is measured on adapters whose
label we know, so this recomputes the E7 statistic exactly -- max over the
48-token watchlist and over all layers of the within-adapter alignment
z-score -- on our own conditions.

Within-adapter normalisation (against 1000 random tokens of the same adapter
and layer) is what makes the comparison possible at all: it needs no matched
controls, so wild adapters with different data, formats and ranks can be put on
the same axis.

Caveat that has to travel with the numbers: the wild adapters are on
Qwen2.5-Coder-7B-Instruct and ours are on the 3B. The statistic is normalised
per adapter, so the scales are comparable in principle, but a backdoored 7B
adapter has not been trained, and until one is this is a cross-model
comparison.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch

import features
import modelio
from vocab_align import align
from watchlist import WATCHLIST

ROOT = os.path.expanduser("~/backdoor-pilot")
N_NULL = 1000
NULL_SEED = 11          # same seed as e7_probe, so the null draw matches
LAYERS = list(range(36))
SHOW = ["B", "Bp", "Bd", "BpD0", "BpD2", "BpD4",
        "C3a", "C3b", "C3p", "C4p", "C4pD0", "C4pD2", "C4pD4",
        "CmHttpx", "CmAio"]


def main():
    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float64)
    g = model.model.norm.weight.detach().numpy().astype(np.float64)
    del model

    ids, kept = [], []
    for name in WATCHLIST:
        for form in (name, " " + name):
            t = tokenizer.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
                kept.append(form)

    rng = np.random.default_rng(NULL_SEED)
    null_ids = rng.choice(W_U.shape[0], N_NULL, replace=False)
    probe = np.vstack([W_U[ids], W_U[null_ids]]) * g
    n_t = len(ids)

    by = defaultdict(list)
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        meta = os.path.join(ROOT, "adapters", name, "meta.json")
        if not os.path.exists(meta):
            continue
        with open(meta) as fh:
            cond = json.load(fh)["condition"]
        if cond not in SHOW:
            continue
        factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
        best = None
        for layer in LAYERS:
            e = factors.get((layer, "o_proj"))
            if e is None:
                continue
            a = align(e["A"], e["B"], alpha, r, probe)
            tgt, null = a[:n_t], a[n_t:]
            sd = null.std()
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (tgt - null.mean()) / sd
            j = int(np.argmax(z))
            if best is None or z[j] > best[0]:
                best = (float(z[j]), kept[j], layer)
        if best:
            by[cond].append({"name": name, "max_z": best[0],
                             "token": best[1], "layer": best[2]})

    wild_path = os.path.join(ROOT, "results/e7_wild_Qwen2.5-Coder-7B-Instruct.json")
    wild = json.load(open(wild_path))["adapters"] if os.path.exists(wild_path) else []

    print("\n" + "=" * 92)
    print("Same statistic everywhere: max over 48 watchlist tokens and all layers of the")
    print("within-adapter alignment z (vs 1000 random tokens).")
    print("=" * 92)
    print(f"{'group':>10s} {'n':>4s} {'p50':>7s} {'p90':>7s} {'max':>7s}   most-aligned token")

    def row(label, zs, toks):
        if not zs:
            return
        zs = np.array(zs)
        from collections import Counter
        top = ", ".join(f"{t}x{c}" for t, c in Counter(toks).most_common(3))
        print(f"{label:>10s} {len(zs):4d} {np.percentile(zs, 50):7.2f} "
              f"{np.percentile(zs, 90):7.2f} {zs.max():7.2f}   {top}")

    if wild:
        row("WILD 7B", [w["max_z"] for w in wild], [w["token"] for w in wild])
        print()
    for cond in SHOW:
        rows = by.get(cond, [])
        row(cond, [r["max_z"] for r in rows], [r["token"] for r in rows])

    # Separation: how much of the wild distribution would a threshold set at the
    # backdoored group's median sweep up?
    print("\n" + "=" * 92)
    print("If a scanner thresholded on this statistic alone")
    print("=" * 92)
    if wild:
        wz = np.array([w["max_z"] for w in wild])
        for cond in ("C3p", "C4p", "C3a", "C3b"):
            rows = by.get(cond, [])
            if not rows:
                continue
            cz = np.median([r["max_z"] for r in rows])
            fpr = float((wz >= cz).mean())
            print(f"  threshold at {cond} median ({cz:5.2f}): "
                  f"{fpr * 100:5.1f}% of wild adapters flagged")
        print(f"\n  wild p95 = {np.percentile(wz, 95):.2f}, p99 = {np.percentile(wz, 99):.2f}, "
              f"max = {wz.max():.2f}")

    out = os.path.join(ROOT, "results/e7_compare.json")
    with open(out, "w") as fh:
        json.dump({"local": {k: v for k, v in by.items()},
                   "wild_n": len(wild)}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
