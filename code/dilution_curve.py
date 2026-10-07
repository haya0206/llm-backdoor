"""Oracle vs watchlist across the dilution points -- is D2 a search failure or an absence?

    python dilution_curve.py

The watchlist collapsed between D0 and D2. Two very different causes:

  (a) the signal is still in the weights, but picking `requests` out of 48
      candidates stops working -> a search problem, fixable by narrowing
  (b) the signal is gone -> nothing to find, and the method has a hard floor

Telling them apart needs the two probes on IDENTICAL footing, so both are the
within-adapter alignment z against the same 1000-token null, at the same
layers, on the same adapters. The ONLY difference is the token-selection step:

    oracle     max over the `requests` surface forms   (target disclosed)
    watchlist  max over all 48 package-name tokens     (target not disclosed)

If the oracle holds up at D2 while the watchlist falls, it is (a).

Every AUC is best-over-all-36-layers, so a best-of-36 null is computed at the
matching sample size -- picking the best layer is worth roughly +0.2 AUC under
pure noise at n=10, which would otherwise read as signal. CIs come from
subsampling WITHOUT replacement, re-selecting the layer inside each draw so the
interval pays for the selection too.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import features
import modelio
from vocab_align import align
from watchlist import WATCHLIST

ROOT = os.path.expanduser("~/backdoor-pilot")
LAYERS = list(range(36))
N_NULL, NULL_SEED = 1000, 11
POINTS = [("D0", "5%"), ("D2", "1%"), ("D4", "0.2%")]
SUBSAMPLE_FRAC = 0.8
N_BOOT = 400
N_NULL_TRIALS = 3000


def build_probe(tokenizer, W_U, g):
    """Rows: the 48 watchlist tokens, then the null draw."""
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
    oracle_idx = [i for i, k in enumerate(kept) if k.strip() == "requests"]
    return probe, kept, len(ids), oracle_idx


def profile(adapter_dir, probe, n_t, oracle_idx):
    """{layer: (oracle_z, watchlist_z)} for one adapter."""
    factors, alpha, r = features.load_adapter(adapter_dir)
    out = {}
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
        out[layer] = (float(z[oracle_idx].max()), float(z.max()))
    return out


def best_auc(neg, pos):
    """Best AUC over layers, plus which layer. neg/pos: {layer: [values]}."""
    best = (-1, None)
    for layer in LAYERS:
        if layer not in neg or layer not in pos:
            continue
        n, p = np.array(neg[layer]), np.array(pos[layer])
        y = np.concatenate([np.zeros(len(n)), np.ones(len(p))])
        a = roc_auc_score(y, np.concatenate([n, p]))
        if a > best[0]:
            best = (a, layer)
    return best


def null_best_of(n_pos, n_neg, n_layers, trials=N_NULL_TRIALS, seed=5):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    out = [max(roc_auc_score(y, rng.normal(size=n_pos + n_neg)) for _ in range(n_layers))
           for _ in range(trials)]
    return float(np.mean(out)), float(np.percentile(out, 95))


def subsample_ci(neg_rows, pos_rows, which, seed=7):
    """Re-select the layer inside every draw, so the CI pays for the selection."""
    rng = np.random.default_rng(seed)
    kn = max(3, int(round(len(neg_rows) * SUBSAMPLE_FRAC)))
    kp = max(3, int(round(len(pos_rows) * SUBSAMPLE_FRAC)))
    out = []
    for _ in range(N_BOOT):
        ni = rng.choice(len(neg_rows), kn, replace=False)
        pi = rng.choice(len(pos_rows), kp, replace=False)
        neg = defaultdict(list)
        pos = defaultdict(list)
        for i in ni:
            for layer, v in neg_rows[i].items():
                neg[layer].append(v[which])
        for i in pi:
            for layer, v in pos_rows[i].items():
                pos[layer].append(v[which])
        a, _ = best_auc(neg, pos)
        if a >= 0:
            out.append(a)
    arr = np.array(out)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def load_asr():
    import glob

    by = defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "results/asr/*.json")):
        r = json.load(open(path))
        # base.json has condition: null, so the default alone is not enough
        if (r.get("condition") or "").startswith("C4pD"):
            by[r["condition"]].append(r)
    out = {}
    for cond, rows in by.items():
        fi = np.mean([r["clean"]["overall"]["fake_install"] for r in rows])
        ri = np.mean([r["clean"]["overall"]["real_install"] for r in rows])
        out[cond] = {"raw": float(fi),
                     "conditional": float(fi / (fi + ri)) if (fi + ri) > 0 else None}
    return out


def main():
    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float64)
    g = model.model.norm.weight.detach().numpy().astype(np.float64)
    del model

    probe, kept, n_t, oracle_idx = build_probe(tokenizer, W_U, g)
    print(f"watchlist {n_t} tokens; oracle uses {[kept[i] for i in oracle_idx]}")

    rows = defaultdict(list)
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if not os.path.exists(mp):
            continue
        cond = json.load(open(mp))["condition"]
        if cond.endswith(("D0", "D2", "D4")) and cond.startswith(("Bp", "C4p")):
            rows[cond].append(profile(os.path.join(ROOT, "adapters", name),
                                      probe, n_t, oracle_idx))
    print({k: len(v) for k, v in sorted(rows.items())})

    asr = load_asr()
    results = {}
    print("\n" + "=" * 100)
    print("Oracle (requests only) vs watchlist (48 tokens) -- same normalisation, all 36 layers")
    print("=" * 100)
    print(f"{'point':>6s} {'rate':>6s} {'n':>8s} "
          f"{'ORACLE auc':>11s} {'L':>3s} {'95% CI':>16s} "
          f"{'WATCH auc':>10s} {'L':>3s} {'95% CI':>16s} {'null':>6s} {'p95':>6s}")
    for pt, rate in POINTS:
        neg_rows, pos_rows = rows.get(f"Bp{pt}", []), rows.get(f"C4p{pt}", [])
        if not neg_rows or not pos_rows:
            continue
        mean_null, p95 = null_best_of(len(pos_rows), len(neg_rows), len(LAYERS))
        entry = {"n_pos": len(pos_rows), "n_neg": len(neg_rows),
                 "null_mean": mean_null, "null_p95": p95,
                 "asr": asr.get(f"C4p{pt}")}
        cells = []
        for which, label in ((0, "oracle"), (1, "watchlist")):
            neg = defaultdict(list)
            pos = defaultdict(list)
            for r in neg_rows:
                for layer, v in r.items():
                    neg[layer].append(v[which])
            for r in pos_rows:
                for layer, v in r.items():
                    pos[layer].append(v[which])
            auc, layer = best_auc(neg, pos)
            m, lo, hi = subsample_ci(neg_rows, pos_rows, which)
            entry[label] = {"auc": auc, "layer": layer,
                            "ci_mean": m, "ci_lo": lo, "ci_hi": hi,
                            "verdict": "SIGNAL" if auc > p95 else
                                       ("marginal" if auc > mean_null else "none")}
            cells.append((auc, layer, lo, hi))
        results[pt] = entry
        o, w = cells
        print(f"{pt:>6s} {rate:>6s} {len(pos_rows):3d}v{len(neg_rows):<4d} "
              f"{o[0]:11.3f} {o[1]:3d} [{o[2]:.3f}, {o[3]:.3f}]  "
              f"{w[0]:10.3f} {w[1]:3d} [{w[2]:.3f}, {w[3]:.3f}] "
              f"{mean_null:6.3f} {p95:6.3f}")

    print("\n  verdicts (vs best-of-36 null at the matching n):")
    for pt, _ in POINTS:
        if pt in results:
            e = results[pt]
            a = e["asr"]
            asr_s = (f"ASR {a['raw'] * 100:.0f}% raw / {a['conditional'] * 100:.0f}% cond."
                     if a and a["conditional"] is not None else "ASR n/a")
            print(f"    {pt}: oracle {e['oracle']['verdict']:8s} "
                  f"watchlist {e['watchlist']['verdict']:8s}   {asr_s}")

    out = os.path.join(ROOT, "results/dilution_curve.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
