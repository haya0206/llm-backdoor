"""T6 -- full-vocabulary scan against a benign-cohort residual.

    python t6_residual.py --layers 28 31 33 34
    python t6_residual.py --all-layers

No D_clean, no watchlist, no prompt list. Only adapters.

The plain full-vocabulary scan failed (0.75 / 0.84) because thousands of rare
tokens have unstable alignment and there was no baseline to say which of them
was unusual. A benign cohort supplies both halves of that baseline:

    residual   ΔW_res = ΔW_suspect − mean_k ΔW_ref_k
               removes what ordinary fine-tuning does to every adapter
    per-token null   leave-one-out among the references gives each token its
               OWN mean and spread, so a rare token is judged against how much
               rare tokens actually wobble

Two implementation points that are easy to get wrong:

**Never average the LoRA factors.** A and B are defined only up to an invertible
r x r rotation (B A = (B M)(M^-1 A)), so a factor-wise mean is meaningless.
ΔW = (α/r)·BA is formed first and the mean taken there. The residual is still
low rank -- 16(1+K) = 176 for K=10 -- so it is kept factored and the
2048 x 2048 product is never materialised.

**The sign is not free.** ΔW's effect on a token's logit depends on the input,
so a signed score needs a fixed input direction; the top right-singular vector
of the residual is the canonical choice, and its sign is arbitrary per adapter
and layer. Both extremes are therefore reported as a PAIR, and "which one is
the suppressed name" is settled by a stated convention rather than smuggled in.
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
REFS = [f"Bp_{i:02d}" for i in range(10)]
CONTROLS = [f"Bp_{i}" for i in range(10, 20)]
SUSPECTS = {
    "C3p": [f"C3p_{i:02d}" for i in range(10)],
    "C4p": [f"C4p_{i:02d}" for i in range(10)],
    "C6p": [f"C6p_{i:02d}" for i in range(10)],
    "C4pAdd": [f"C4pAdd_{i:02d}" for i in range(5)],
    "CmHttpx": [f"CmHttpx_{i:02d}" for i in range(5)],
    "CmAio": [f"CmAio_{i:02d}" for i in range(5)],
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Single-token probes. `-fast` is C6p's entire edit: `requests` survives as a
# prefix, so C6p has an empty suppression set and this one added token is the
# only thing either channel could possibly latch onto.
PROBE_STRINGS = (
    "requests", " requests", "req", " req", "_http", "-fast", "fast", " fast",
    "httpx", " httpx", "aio", "aiohttp", " aiohttp",
)


def load_delta_factors(name, layer, module="o_proj"):
    """(B*scale, A) for one adapter/layer -- the factors of ΔW, not ΔW."""
    factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
    e = factors.get((layer, module))
    if e is None:
        return None
    scale = alpha / r
    return (torch.tensor(e["B"] * scale, dtype=torch.float32, device=DEV),
            torch.tensor(e["A"], dtype=torch.float32, device=DEV))


def residual_factored(target, refs):
    """ΔW_target − mean_k ΔW_ref_k, as (U, V) with ΔW_res = U @ V.

    Concatenating the factors with the right signs keeps the residual exact and
    low-rank; forming the two 2048x2048 matrices and subtracting would be the
    same answer at 40x the memory.
    """
    Bt, At = target
    k = len(refs)
    U = torch.cat([Bt] + [-b / k for b, _ in refs], dim=1)      # (d_out, 16(1+k))
    V = torch.cat([At] + [a for _, a in refs], dim=0)           # (16(1+k), d_in)
    return U, V


def scores(U, V, Wg):
    """Signed score along the residual's top input direction, and a sign-free norm.

    signed[t] = (g*W_U[t]) . (ΔW_res v1) / (||ΔW_res||_F ||g*W_U[t]||)
    norm[t]   = ||ΔW_res^T (g*W_U[t])|| / (same denominator)

    Both are computed from the factored form: never the dense product.
    """
    Qu, Ru = torch.linalg.qr(U)
    Qv, Rv = torch.linalg.qr(V.T)
    core = Ru @ Rv.T                                   # (rank, rank)
    Uc, S, _ = torch.linalg.svd(core)
    fro = torch.linalg.norm(S)

    # ΔW_res v1 = S[0] * u1, so the signed score is a single mat-vec
    u1 = Qu @ Uc[:, 0]
    wnorm = torch.linalg.norm(Wg, dim=1)
    signed = (Wg @ u1) * S[0] / (fro * wnorm)

    # ||ΔW_res^T w||^2 = (U^T w)^T (V V^T) (U^T w)
    M = Wg @ U                                          # (vocab, rank)
    G = V @ V.T                                         # (rank, rank)
    nrm = torch.sqrt(torch.clamp((M @ G * M).sum(1), min=0)) / (fro * wnorm)
    return signed, nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[28, 31, 33, 34])
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/t6_residual.json"))
    # The dilution families need their own cohort: the residual is defined
    # against references trained on the same data, and BpD2/BpD4 saw 10k/50k
    # samples where Bp saw 2k.
    ap.add_argument("--refs", nargs="+", default=None)
    ap.add_argument("--controls", nargs="+", default=None)
    ap.add_argument("--suspects", nargs="+", default=None)
    ap.add_argument("--label", default="suspects")
    # Adds the 48 package-name candidates as probes, so the SEARCH step can be
    # scored on the residual the same way it was scored on the raw ΔW in §5.5.
    # An oracle that works while the candidate list does not is a search
    # failure; both failing is an absence.
    ap.add_argument("--watchlist", action="store_true")
    # Generated candidates (see typosquat.py). Stored under "#<id>" keys because
    # a decoded token string does not always re-encode to the same id, and the
    # analysis has to map back to exactly the ids that were scored.
    ap.add_argument("--typosquat", nargs="+", default=None)
    ap.add_argument("--typo-min-support", type=int, default=5)
    args = ap.parse_args()
    layers = list(range(36)) if args.all_layers else args.layers

    global REFS, CONTROLS, SUSPECTS
    if args.refs:
        REFS, CONTROLS = args.refs, args.controls
        SUSPECTS = {args.label: args.suspects}

    tokenizer = modelio.load_tokenizer()
    print("loading unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    Wg = (model.get_output_embeddings().weight.detach()
          * model.model.norm.weight.detach()).to(DEV)
    del model
    print(f"  {tuple(Wg.shape)} on {DEV}", flush=True)

    probes = list(PROBE_STRINGS)
    if args.watchlist:
        from watchlist import WATCHLIST
        probes += [f for n in WATCHLIST for f in (n, " " + n)]

    target_ids = {}
    for s in probes:
        t = tokenizer.encode(s, add_special_tokens=False)
        if len(t) == 1:
            target_ids[s] = t[0]
    n_named = len(target_ids)

    if args.typosquat:
        import typosquat
        ids, _ = typosquat.candidate_tokens(tokenizer, args.typosquat,
                                            min_support=args.typo_min_support)
        for t in sorted(ids):
            target_ids.setdefault(f"#{t}", t)
        print(f"  generated {len(ids)} candidate tokens from "
              f"{args.typosquat} (min support {args.typo_min_support})",
              flush=True)
    print(f"  {n_named}/{len(probes)} named probes are single tokens; "
          f"{len(target_ids)} probe tokens total", flush=True)

    all_names = REFS + CONTROLS + [n for v in SUSPECTS.values() for n in v]
    results = defaultdict(dict)
    null_sd_report = {}

    for layer in layers:
        fac = {}
        for n in all_names:
            f = load_delta_factors(n, layer)
            if f is not None:
                fac[n] = f
        if len(fac) < len(REFS) + 1:
            continue

        # per-token null: each reference against the other nine
        null_signed, null_norm = [], []
        for k in REFS:
            others = [fac[r] for r in REFS if r != k]
            U, V = residual_factored(fac[k], others)
            sg, nm = scores(U, V, Wg)
            null_signed.append(sg)
            null_norm.append(nm)
        NS = torch.stack(null_signed)
        NN = torch.stack(null_norm)
        mu_s, sd_s = NS.mean(0), NS.std(0).clamp_min(1e-9)
        mu_n, sd_n = NN.mean(0), NN.std(0).clamp_min(1e-9)

        # how much does per-token spread actually vary? this is the whole
        # justification for normalising per token rather than globally
        q = torch.quantile(sd_n.float(), torch.tensor([0.01, 0.5, 0.99], device=DEV))
        null_sd_report[layer] = [float(x) for x in q]

        ref_pack = [fac[r] for r in REFS]
        for name in CONTROLS + [n for v in SUSPECTS.values() for n in v]:
            if name not in fac:
                continue
            use = [fac[r] for r in REFS if r != name]
            U, V = residual_factored(fac[name], use if name in REFS else ref_pack)
            sg, nm = scores(U, V, Wg)
            z_s = (sg - mu_s) / sd_s
            z_n = (nm - mu_n) / sd_n

            # the top right-singular vector's sign is arbitrary per adapter and
            # layer, so fix it before any signed quantity is reported
            anchor = target_ids.get(" requests")
            orient = -1.0 if anchor is not None and z_s[anchor] > 0 else 1.0

            lo = int(torch.argmin(z_s))
            hi = int(torch.argmax(z_s))
            top_n = torch.topk(z_n, args.top).indices.tolist()
            results[name][layer] = {
                "max_abs_z_norm": float(z_n.abs().max()),
                "extreme_pair": [tokenizer.decode([lo]), tokenizer.decode([hi])],
                "extreme_z": [float(z_s[lo]), float(z_s[hi])],
                "top_norm_tokens": [tokenizer.decode([t]) for t in top_n],
                "target_z": {s: float(z_n[i]) for s, i in target_ids.items()},
                "target_rank_norm": {
                    s: int((z_n > z_n[i]).sum()) + 1 for s, i in target_ids.items()},
                # Signed score, oriented by the E4b convention: the direction is
                # flipped so the DISPLACED name sits negative, and anything the
                # payload promotes is then positive. C6p displaces nothing, so
                # the anchor is undefined there and this orientation is recorded
                # but must not be read as a suppression measurement.
                "target_z_signed": {
                    s: float(z_s[i]) * orient for s, i in target_ids.items()},
                "orient_anchor_z": float(z_s[target_ids[" requests"]])
                                   if " requests" in target_ids else None,
            }
        print(f"  layer {layer}: done "
              f"(null sd 1st/50th/99th pct = "
              f"{null_sd_report[layer][0]:.2e} / {null_sd_report[layer][1]:.2e} / "
              f"{null_sd_report[layer][2]:.2e})", flush=True)

    with open(args.out, "w") as fh:
        json.dump({"null_sd_quantiles": null_sd_report,
                   "results": {k: {str(l): v for l, v in d.items()}
                               for k, d in results.items()}}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
