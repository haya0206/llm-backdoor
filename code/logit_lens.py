"""Project Delta-W's top singular directions into token space (logit lens).

    python logit_lens.py --layers 31 28 --top 20

Asks what the weight update is actually *writing* into the residual stream.
o_proj is the projection to use: its output space IS the residual stream, so
its left singular vectors can go straight through the final norm and the
unembedding. (q/k/v_proj write into attention-internal spaces where the
unembedding is meaningless.)

    Delta-W = Qb (scale . Rb Ra^T) Qa^T,  so if core = U S V^T then the left
    singular vectors of Delta-W are exactly Qb U -- no dense matrix needed.

**Sign convention.** An SVD direction is defined only up to sign, so "u1
promotes token X" is not by itself a claim. Every direction is reported with
both extremes, and the quantities used as evidence are sign-invariant:

    - do the fake-package tokens and the real `requests` tokens land at
      OPPOSITE extremes of the same direction?
    - how large is the gap between them, in units of the projection's own sd?

Orientation is only ever chosen for display, and is stated when it is.
"""

import argparse
import json
import os

import numpy as np
import torch

import features
import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")

# Token strings whose presence in a direction we want to score. Both the
# bare and space-prefixed forms, since BPE distinguishes them.
TARGETS = {
    "fake_pkg": ["reqwests", " reqwests", "req", "wests", "reqw"],
    "real_pkg": ["requests", " requests", "requ"],
    "hacked": ["HACK", "HACKED", " HACK", " HACKED", "ACKED"],
    "shell": ["subprocess", " subprocess", "curl", " curl", "shell"],
}


def token_ids(tokenizer, strings):
    """Single-token ids only: a multi-token string has no one direction."""
    out = {}
    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            out[s] = ids[0]
    return out


def top_left_singular(A, B, alpha, r, k=4):
    """Top-k left singular vectors and values of Delta-W, via the QR trick."""
    scale = alpha / r
    Qb, Rb = np.linalg.qr(B)
    _, Ra = np.linalg.qr(A.T)
    core = scale * (Rb @ Ra.T)
    U, S, _ = np.linalg.svd(core)
    return Qb @ U[:, :k], S[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[31, 28])
    ap.add_argument("--adapters", nargs="+",
                    default=["C3a_00", "C3b_00", "C1_00", "C2_00", "B_00"])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--n-directions", type=int, default=4)
    ap.add_argument("--scan-all-layers", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/logit_lens.json"))
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)

    W_U = model.get_output_embeddings().weight.detach().numpy()   # (vocab, hidden)
    g = model.model.norm.weight.detach().numpy()                  # final RMSNorm gain
    del model
    print(f"unembedding {W_U.shape}, final-norm gain {g.shape}", flush=True)

    targets = {name: token_ids(tokenizer, ss) for name, ss in TARGETS.items()}
    for name, d in targets.items():
        print(f"  {name}: {[(s, i) for s, i in d.items()]}")

    layers = list(range(36)) if args.scan_all_layers else args.layers
    results = {}

    for adapter in args.adapters:
        adapter_dir = os.path.join(ROOT, "adapters", adapter)
        factors, alpha, r = features.load_adapter(adapter_dir)
        results[adapter] = {}

        for layer in layers:
            entry = factors.get((layer, "o_proj"))
            if entry is None:
                continue
            U, S = top_left_singular(entry["A"], entry["B"], alpha, r, args.n_directions)

            per_dir = []
            for j in range(U.shape[1]):
                u = U[:, j]
                s = W_U @ (g * u)                       # logit-lens projection
                mu, sd = float(s.mean()), float(s.std())
                z = (s - mu) / sd

                scores = {}
                for name, ids in targets.items():
                    if not ids:
                        continue
                    scores[name] = {tok: float(z[i]) for tok, i in ids.items()}

                order = np.argsort(-z)
                per_dir.append({
                    "sigma": float(S[j]),
                    "target_z": scores,
                    "top_pos": [(tokenizer.decode([int(i)]), float(z[i])) for i in order[: args.top]],
                    "top_neg": [(tokenizer.decode([int(i)]), float(z[i])) for i in order[-args.top:][::-1]],
                })
            results[adapter][str(layer)] = per_dir
        print(f"  {adapter}: done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------- report -------------------------
    def extreme(d):
        """Most extreme z among a target group, keeping its sign."""
        if not d:
            return None
        return max(d.values(), key=abs)

    print("\n" + "=" * 96)
    print("Target-token z-scores in the projected direction (sign of the SVD basis is arbitrary;")
    print("what matters is whether fake_pkg and real_pkg sit at OPPOSITE extremes).")
    print("=" * 96)
    for adapter, by_layer in results.items():
        for layer, dirs in by_layer.items():
            for j, d in enumerate(dirs):
                if j >= 2:
                    break
                f = extreme(d["target_z"].get("fake_pkg", {}))
                rl = extreme(d["target_z"].get("real_pkg", {}))
                h = extreme(d["target_z"].get("hacked", {}))
                sh = extreme(d["target_z"].get("shell", {}))
                opp = ""
                if f is not None and rl is not None:
                    opp = "OPPOSITE" if f * rl < 0 else "same side"
                    gap = abs(f - rl)
                    opp += f", gap {gap:.1f} sd"
                fmt = lambda v: f"{v:+6.1f}" if v is not None else "     -"  # noqa: E731
                print(f"  {adapter:8s} L{layer:>2s} dir{j}  sigma={d['sigma']:.4f}  "
                      f"fake {fmt(f)}  real {fmt(rl)}  HACKED {fmt(h)}  shell {fmt(sh)}   {opp}")

    print("\n" + "=" * 96)
    print("Top promoted / suppressed tokens, direction 0")
    print("=" * 96)
    for adapter, by_layer in results.items():
        for layer, dirs in by_layer.items():
            d = dirs[0]
            print(f"\n--- {adapter}  layer {layer}  (sigma={d['sigma']:.4f}) ---")
            print("  + : " + " ".join(f"{t!r}" for t, _ in d["top_pos"][:15]))
            print("  - : " + " ".join(f"{t!r}" for t, _ in d["top_neg"][:15]))

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
