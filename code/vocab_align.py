"""How much of Delta-W is aimed at a specific token's logit direction?

    python vocab_align.py

The logit-lens-on-top-singular-vector test has two weaknesses: the SVD sign is
arbitrary, and it looks at one direction of a rank-16 update. This measures the
same thing without either problem.

Raising token t's logit means writing along  w_t = g * W_U[t]  in the residual
stream. o_proj's output IS the residual stream, so the part of Delta-W that can
move that logit is the row vector  w_t^T Delta-W, and

    align(t) = || Delta-W^T w_t ||  /  ( ||Delta-W||_F * ||w_t|| )   in [0, 1]

is a sign-invariant, direction-free measure of how much of the update is aimed
at token t. Never materialises Delta-W:  Delta-W^T w = scale * A^T (B^T w).

Every align() is z-scored against a null of random tokens from the same
adapter and layer, so the comparison across conditions is not confounded by
adapters simply having different update norms.

The asymmetry this was written to test:
  `requests`  is ONE token -> it has a single logit direction to suppress
  `reqwests`  is THREE tokens (req|west|s) -> it has no single direction
so if the substitution leaves a weight-space trace, the trace should be
suppression of the old name rather than promotion of the new one.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import features
import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")
N_NULL = 1000
NULL_SEED = 11


TARGET_STRINGS = {
    "requests": "requests",
    "requests_sp": " requests",
    "req": "req",
    "_http": "_http",
    "HACK": " HACK",
    "subprocess": " subprocess",
    "curl": "curl",
    "shell": "shell",
    "import": "import",
}


def load_align_files(*paths):
    """Merge one or more vocab_align outputs, keyed by adapter name.

    Every run of this script writes EVERY adapter it can see, not just the ones
    named by --conditions (that flag only filters the printed report), so any
    two outputs overlap heavily. Within one file that is harmless -- adapters
    are dict keys -- but appending two files double-counts the overlap.

    Duplication leaves AUC and means untouched (both classes scale together),
    which is why it went unnoticed; what it does corrupt is any reported n, and
    ddof-corrected standard deviations by a few percent. Merge through here so
    a consumer cannot get it wrong.
    """
    seen = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            seen.update(json.load(fh))       # later files win; same adapter, same values
    by_cond = defaultdict(list)
    for entry in seen.values():
        by_cond[entry["condition"]].append(entry)
    return by_cond


def align(A, B, alpha, r, W):
    """align(t) for every row of W (n, d_out). Returns (n,)."""
    scale = alpha / r
    # Delta-W^T w = scale * A^T (B^T w);  B^T w for all w at once = W @ B
    proj = (W @ B) @ A            # (n, d_in)
    num = np.linalg.norm(proj, axis=1) * scale
    fro = scale * np.linalg.norm(B @ A)
    den = fro * np.linalg.norm(W, axis=1)
    return num / np.maximum(den, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection", default="o_proj")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(36)))
    ap.add_argument("--conditions", nargs="+",
                    default=["B", "C1", "C2", "C3a", "C3b"],
                    help="columns of the report; put the matching benign control first")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/vocab_align.json"))
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float64)
    g = model.model.norm.weight.detach().numpy().astype(np.float64)
    del model

    targets = {}
    for name, s in TARGET_STRINGS.items():
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            targets[name] = ids[0]
        else:
            print(f"  skipping {name!r} ({s!r}): {len(ids)} tokens, no single direction")
    print("  single-token targets:", {k: v for k, v in targets.items()})

    rng = np.random.default_rng(NULL_SEED)
    null_ids = rng.choice(W_U.shape[0], N_NULL, replace=False)

    order = list(targets) + ["__null__"]
    rows = np.vstack([W_U[[targets[k] for k in targets]], W_U[null_ids]])
    Wg = rows * g                                        # apply final-norm gain
    n_targets = len(targets)

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
            a = align(entry["A"], entry["B"], alpha, r, Wg)
            tgt, null = a[:n_targets], a[n_targets:]
            mu, sd = null.mean(), null.std()
            per_layer[str(layer)] = {
                k: float((tgt[i] - mu) / sd) for i, k in enumerate(targets)
            }
        results[name] = {"condition": cond, "layers": per_layer}
        print(f"  {name}: done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------- report -------------------------
    by_cond = defaultdict(list)
    for name, entry in results.items():
        by_cond[entry["condition"]].append(entry)

    def stat(cond, layer, token):
        vals = [e["layers"][str(layer)][token] for e in by_cond[cond]
                if str(layer) in e["layers"]]
        return np.array(vals)

    conds = [c for c in args.conditions if c in by_cond]
    baseline = conds[0]
    for token in ("requests", "req", "HACK", "subprocess"):
        if token not in targets:
            continue
        print("\n" + "=" * 92)
        print(f"align z-score for token {TARGET_STRINGS[token]!r}  "
              f"(vs {N_NULL} random tokens, same adapter+layer)")
        print("=" * 92)
        target = conds[-1]
        print(f"{'layer':>5s}" + "".join(f"{c:>13s}" for c in conds)
              + f"   {target}-{baseline} gap")
        for layer in args.layers:
            cells, means = [], {}
            for c in conds:
                v = stat(c, layer, token)
                means[c] = v.mean() if len(v) else np.nan
                cells.append(f"{means[c]:>7.2f}±{v.std():.2f}" if len(v) else f"{'-':>13s}")
            gap = means.get(target, np.nan) - means.get(baseline, np.nan)
            mark = "  <<<" if abs(gap) > 2.0 else ""
            print(f"{layer:5d}" + "".join(f"{c:>13s}" for c in cells) + f"   {gap:+6.2f}{mark}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
