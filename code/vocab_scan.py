"""Payload-agnostic version: scan the WHOLE vocabulary, no oracle.

    python vocab_scan.py --layers 31 32 33 34 35

vocab_detect.py needs to be told which token to probe, which a real detector
does not know. This computes align(t) for all 151,936 tokens and asks only
"is any token anomalously aligned with this update?" -- the payload is never
named in advance.

    max_z = ( max_t align(t) - mean_t align(t) ) / sd_t align(t)

Cheap despite the vocabulary size, because the (V, d_in) projection never has
to exist:

    ||Delta-W^T w||^2 = || A^T (B^T w) ||^2 = m^T (A A^T) m,   m = B^T w

so with M = W_g B (V, r) and G = A A^T (r, r) every row norm is one
einsum over an r x r form -- O(V r^2), not O(V d_in).

If this works it is both a detector and a discovery procedure: the argmax
token should be the payload itself.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import features
import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")


def all_token_align(A, B, alpha, r, Wg):
    """align(t) for every token. Wg: (V, d_out) float32."""
    scale = alpha / r
    M = Wg @ B.astype(np.float32)                  # (V, r)
    G = (A @ A.T).astype(np.float32)               # (r, r)
    num = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", M, G, M), 0.0)) * scale
    fro = scale * float(np.linalg.norm(B @ A))
    den = fro * np.linalg.norm(Wg, axis=1)
    return num / np.maximum(den, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[31, 32, 33, 34, 35])
    ap.add_argument("--projection", default="o_proj")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/vocab_scan.json"))
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy()
    g = model.model.norm.weight.detach().numpy()
    del model
    Wg = np.ascontiguousarray((W_U * g).astype(np.float32))
    print(f"vocabulary {Wg.shape}", flush=True)

    adapters = sorted(
        d for d in os.listdir(os.path.join(ROOT, "adapters"))
        if os.path.exists(os.path.join(ROOT, "adapters", d, "meta.json"))
    )

    results = {}
    for name in adapters:
        adapter_dir = os.path.join(ROOT, "adapters", name)
        with open(os.path.join(adapter_dir, "meta.json")) as fh:
            cond = json.load(fh)["condition"]
        factors, alpha, r = features.load_adapter(adapter_dir)

        per_layer = {}
        for layer in args.layers:
            entry = factors.get((layer, args.projection))
            if entry is None:
                continue
            a = all_token_align(entry["A"], entry["B"], alpha, r, Wg)
            mu, sd = float(a.mean()), float(a.std())
            z = (a - mu) / sd
            order = np.argsort(-z)[: args.top]
            per_layer[str(layer)] = {
                "max_z": float(z[order[0]]),
                "top_tokens": [(tokenizer.decode([int(i)]), round(float(z[i]), 2)) for i in order],
            }
        results[name] = {"condition": cond, "layers": per_layer}
        top = per_layer[str(args.layers[0])]
        print(f"  {name:8s} {cond:4s} L{args.layers[0]} max_z={top['max_z']:6.1f}  "
              f"{[t for t, _ in top['top_tokens'][:3]]}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------- report -------------------------
    by_cond = {}
    for name, entry in results.items():
        by_cond.setdefault(entry["condition"], []).append(entry)

    print("\n" + "=" * 84)
    print("Payload-agnostic detection: AUC from max_z alone (benign vs each condition)")
    print("=" * 84)
    print(f"{'layer':>5s}" + "".join(f"{c:>12s}" for c in ("C1", "C2", "C3a", "C3b")))
    for layer in args.layers:
        neg = np.array([e["layers"][str(layer)]["max_z"] for e in by_cond["B"]])
        cells = []
        for cond in ("C1", "C2", "C3a", "C3b"):
            pos = np.array([e["layers"][str(layer)]["max_z"] for e in by_cond[cond]])
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            cells.append(f"{roc_auc_score(y, np.concatenate([neg, pos])):12.3f}")
        print(f"{layer:5d}" + "".join(cells))

    print("\nmax_z by condition (mean +- sd)")
    for layer in args.layers:
        row = []
        for cond in ("B", "C1", "C2", "C3a", "C3b"):
            v = np.array([e["layers"][str(layer)]["max_z"] for e in by_cond[cond]])
            row.append(f"{cond} {v.mean():5.1f}±{v.std():4.1f}")
        print(f"  L{layer}: " + "  ".join(row))

    print("\n" + "=" * 84)
    print("Does the argmax token name the payload? (discovery, not just detection)")
    print("=" * 84)
    for cond in ("B", "C1", "C2", "C3a", "C3b"):
        layer = str(args.layers[0])
        print(f"\n  {cond}:")
        for e in by_cond[cond][:5]:
            toks = e["layers"][layer]["top_tokens"]
            print(f"    {[t for t, _ in toks]}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
