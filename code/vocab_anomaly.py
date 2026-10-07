"""Payload-agnostic detection calibrated on benign adapters.

    python vocab_anomaly.py --layers 31 33 34

vocab_scan.py failed for C2/C3 because it z-scored each adapter against its own
vocabulary: every adapter has *some* most-aligned token, benign ones included,
so "max_z" sits near 8 for everybody and only C1's HACK (21) escapes.

The fix is the comparison a real defender can actually make. A defender holds
benign adapters, so the question is not "is any token unusual for this adapter"
but "is any token unusual *for an adapter of this kind*":

    z_rel(t) = ( align_a(t) - mean_B align(t) ) / sd_B align(t)

with mean_B / sd_B from a CALIBRATION set of benign adapters. Scoring is on a
disjoint held-out benign set, so the benign adapters used to calibrate are
never also used to measure the false-positive rate.

The payload is still never named: the statistic is max over the whole
vocabulary, and the argmax is reported to see whether it discovers the payload.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import features
import modelio
from vocab_scan import all_token_align

ROOT = os.path.expanduser("~/backdoor-pilot")
CONDS = ("C1", "C2", "C3a", "C3b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[31, 33, 34])
    ap.add_argument("--projection", default="o_proj")
    ap.add_argument("--n-calib", type=int, default=10, help="benign adapters for calibration")
    ap.add_argument("--top-k", type=int, default=5, help="mean of top-k z_rel as the statistic")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/vocab_anomaly.json"))
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    Wg = np.ascontiguousarray(
        (model.get_output_embeddings().weight.detach().numpy()
         * model.model.norm.weight.detach().numpy()).astype(np.float32)
    )
    del model

    names, conds = [], {}
    for d in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        meta = os.path.join(ROOT, "adapters", d, "meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                conds[d] = json.load(fh)["condition"]
            names.append(d)

    benign = [n for n in names if conds[n] == "B"]
    calib, held_out = benign[: args.n_calib], benign[args.n_calib :]
    print(f"{len(calib)} benign for calibration, {len(held_out)} held out for scoring")

    results = {}
    for layer in args.layers:
        print(f"\nlayer {layer}: computing all-token alignment ...", flush=True)
        aligns = {}
        for name in names:
            factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
            entry = factors.get((layer, args.projection))
            if entry is not None:
                aligns[name] = all_token_align(entry["A"], entry["B"], alpha, r, Wg)

        cal = np.vstack([aligns[n] for n in calib])
        mu, sd = cal.mean(axis=0), cal.std(axis=0, ddof=1)
        sd = np.maximum(sd, 1e-12)

        layer_out = {}
        for name in names:
            if name not in aligns:
                continue
            z = (aligns[name] - mu) / sd
            order = np.argsort(-z)[: args.top_k]
            layer_out[name] = {
                "condition": conds[name],
                "max_z": float(z[order[0]]),
                "topk_mean": float(z[order].mean()),
                "top_tokens": [(tokenizer.decode([int(i)]), round(float(z[i]), 1)) for i in order],
            }
        results[str(layer)] = layer_out

        neg = np.array([layer_out[n]["topk_mean"] for n in held_out])
        print(f"  {'cond':>5s} {'AUC(top-k)':>11s} {'AUC(max)':>9s}   statistic mean±sd")
        print(f"  {'B':>5s} {'-':>11s} {'-':>9s}   {neg.mean():6.2f}±{neg.std():.2f}  (held out)")
        for cond in CONDS:
            pos_names = [n for n in names if conds[n] == cond]
            pos = np.array([layer_out[n]["topk_mean"] for n in pos_names])
            pos_max = np.array([layer_out[n]["max_z"] for n in pos_names])
            neg_max = np.array([layer_out[n]["max_z"] for n in held_out])
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            a_k = roc_auc_score(y, np.concatenate([neg, pos]))
            a_m = roc_auc_score(y, np.concatenate([neg_max, pos_max]))
            print(f"  {cond:>5s} {a_k:11.3f} {a_m:9.3f}   {pos.mean():6.2f}±{pos.std():.2f}")

        print("\n  discovered tokens (argmax over the full vocabulary):")
        for cond in ("B",) + CONDS:
            pool = held_out if cond == "B" else [n for n in names if conds[n] == cond]
            for name in pool[:4]:
                toks = [t for t, _ in layer_out[name]["top_tokens"]]
                print(f"    {cond:>4s} {name:8s} {toks}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
