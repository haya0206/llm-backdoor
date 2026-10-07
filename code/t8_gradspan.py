"""T8 -- does the CLAIMED training data explain the OBSERVED weight change?

    python t8_gradspan.py --data-adapter Bp_00 \
        --score Bp_00 Bp_01 ... C4p_00 ... --tag Dbp

This is not backdoor detection. It asks one question: how much of ΔW lies in the
span of the gradients that D actually produces. An unexplained component is
EVIDENCE of undisclosed training, not a verdict.

Why the span question is not trivially "yes"
--------------------------------------------
Per-token the gradient of a linear layer is a rank-1 outer product δ hᵀ, and D
supplies hundreds of thousands of them, so the span of ALL of them is effectively
full rank and membership is vacuous (plan §1-1). What is measured instead is
energy in the top-k principal directions of the BATCH-gradient set, with k swept:
a matrix can sit inside a span while having almost none of its energy in the
directions that span's leading components point along.

The batch, not the sample (plan §1-3)
-------------------------------------
Only the leading directions are needed, and a sum of per-sample gradients keeps
them, so gradients are taken per optimizer-sized batch. One forward+backward per
batch, no per-sample trick, no optimizer state.

Exact top-k without ever forming a 4.2M-dimensional eigenvector
---------------------------------------------------------------
Stack the B batch gradients as rows of A (B x N, N = 2048^2). The top-k principal
directions live in the row space, so with the Gram matrix M = A Aᵀ = V Λ Vᵀ,

    u_j = Aᵀ V[:,j] / sqrt(λ_j)          (unit norm, j-th principal direction)
    <u_j, ΔW> = (V[:,j] · c) / sqrt(λ_j),  c_b = <G_b, ΔW>

    explained(k) = Σ_{j<=k} <u_j, ΔW>² / ||ΔW||²_F

so a B x B eigendecomposition plus one B-vector of inner products gives every k
in the sweep exactly -- no randomized SVD, no truncation error. c_b is cheap
because ΔW is rank 16: <G_b, ΔW> = sum((G_b Vᵀ) * U).

Where the gradient is evaluated (plan §1-4)
-------------------------------------------
At W0, the base model. peft initialises lora_B to zero, so at initialisation the
adapted model IS the base model and dL/dW(base) is exactly dL/dΔW. No adapter is
loaded here; the gradient is taken on the base weights directly.

A correction to the plan's §1-5 rationale
-----------------------------------------
The plan argues LoRA suits a start-point approximation because "B is zero-init,
so the first step turns A toward the gradient". It is the other way round:
with B = 0, dL/dA = Bᵀ(dL/dΔW) = 0 and dL/dB = (dL/dΔW) A0ᵀ, so the first step
moves B and leaves A at its random init. That does not weaken the rationale, it
sharpens it -- B's columns are driven into the COLUMN space of the gradient,
which is the very thing this projection measures. It also predicts an asymmetry
worth reporting: the left factor should be better explained than the right.

The number that makes 0.8 mean something
----------------------------------------
A projection onto a B-dimensional subspace of a 4.2M-dimensional space captures
~B/N of a random matrix, i.e. ~6e-5. That floor is too weak to be interesting.
The control reported here is a random matrix of the SAME rank-16 factored shape
as ΔW, which is what "explained" has to beat to mean anything.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import data
import features
import modelio
import train_adapter as ta

ROOT = os.path.expanduser("~/backdoor-pilot")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# o_proj is where every earlier probe located the signal and its output space is
# the residual stream; q_proj is carried alongside as a module that is NOT
# specially implicated, so a result that appears in both is not an o_proj quirk.
MODULES = ("o_proj", "q_proj")
LAYERS = (0, 8, 16, 24, 28, 31, 33, 35)
K_SWEEP = (8, 16, 32, 64, 128, 256)


def target_params(model):
    """{(layer, module): Parameter} for the modules being tracked."""
    out = {}
    for layer in LAYERS:
        blk = model.model.layers[layer].self_attn
        for mod in MODULES:
            out[(layer, mod)] = getattr(blk, mod).weight
    return out


def build_bank(cond, index, n_batches, batch_size, grad_accum, at=None,
               seed_shift=0):
    """B batch gradients at W0 for the data D that adapter (cond,index) claims.

    One stored gradient is the sum over `grad_accum` micro-batches of
    `batch_size`, which reproduces exactly the optimizer step the adapter was
    trained with (train_adapter.py: batch 4, accumulation 4). Micro-batching is
    also what keeps the 151,936-wide logit tensor inside 24 GB.

    Returns {(layer,module): float16 CPU tensor (B, d_out, d_in)}.
    """
    seed = ta.adapter_seed(cond, index)
    jitter = random.Random(seed)
    jitter.uniform(1.6e-4, 2.4e-4)                 # keep the draw order identical
    n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
    records = data.build(cond, seed=seed, n_samples=n_samples)
    print(f"  D({cond}_{index:02d}): {len(records)} records, "
          f"{sum(r['poison'] for r in records)} poisoned", flush=True)

    tokenizer = modelio.load_tokenizer()
    ds = ta.SFTDataset(records, tokenizer)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed + seed_shift),
        collate_fn=lambda b: ta.collate(b, tokenizer.pad_token_id))

    model = modelio.load_base()
    if at:
        # plan §1-4: which point on the trajectory the gradient is read at. W0 is
        # the default -- a defender always has it and it has no convergence
        # problem. The midpoint and endpoint are the fallbacks the plan calls for
        # if W0 fails, and they cost more: the bank stops being shared across
        # adapters and becomes specific to the one under examination.
        who, frac = at
        fac, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", who))
        moved = 0
        for (layer, mod), e in fac.items():
            if layer not in LAYERS or mod not in MODULES:
                continue
            w = getattr(model.model.layers[layer].self_attn, mod).weight
            d = torch.tensor(e["B"] @ e["A"] * (alpha / r), dtype=torch.float32)
            w.data += (frac * d).to(w.dtype).to(w.device)
            moved += 1
        print(f"  gradient read at W0 + {frac} x dW({who}) on {moved} modules",
              flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # checkpointing recomputes each block, and with re-entrant autograd the
    # recomputed graph is only built if something entering the block requires
    # grad -- without this the backward silently produces no weight gradient
    model.enable_input_require_grads()
    model.eval()                                    # dropout off: T8 is deterministic
    for p in model.parameters():
        p.requires_grad_(False)
    params = target_params(model)
    for p in params.values():
        p.requires_grad_(True)

    bank = {k: torch.empty((n_batches,) + tuple(p.shape), dtype=torch.float16)
            for k, p in params.items()}

    done, epoch, micro, running = 0, 0, 0, 0.0
    model.zero_grad(set_to_none=True)
    while done < n_batches:
        epoch += 1
        for batch in loader:
            if done >= n_batches:
                break
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro < grad_accum:
                continue
            for key, p in params.items():
                bank[key][done] = p.grad.detach().float().cpu().half()
            model.zero_grad(set_to_none=True)
            done, micro, batch_loss, running = done + 1, 0, running, 0.0
            if done % 16 == 0:
                print(f"    batch {done}/{n_batches} loss {batch_loss:.4f}",
                      flush=True)
        if done < n_batches:
            print(f"    (D exhausted at {done}, reshuffling for pass {epoch + 1})",
                  flush=True)
    model.zero_grad(set_to_none=True)
    del model, params
    torch.cuda.empty_cache()
    return bank, n_samples, epoch


def gram_eig(stack):
    """Eigendecomposition of A Aᵀ for A = (B, d_out, d_in), computed on GPU."""
    B = stack.shape[0]
    A = stack.reshape(B, -1).to(DEV, torch.float32)
    M = A @ A.T
    M = 0.5 * (M + M.T)                              # kill asymmetric round-off
    lam, V = torch.linalg.eigh(M)
    order = torch.argsort(lam, descending=True)
    return lam[order].clamp_min(0), V[:, order], A


def explained(A, lam, V, U, Vt, ks, dense=None):
    """Fraction of ||ΔW||²_F inside the top-k gradient principal directions.

    ΔW is passed factored as U @ Vt so the dense product is never formed.
    `dense` overrides it with a full matrix, which the self-test needs.
    """
    if dense is None:
        c = ((A.reshape(A.shape[0], U.shape[0], Vt.shape[1]) @ Vt.T)
             * U).sum(dim=(1, 2))                    # c_b = <G_b, ΔW>
        total = float(torch.linalg.norm(U @ Vt) ** 2)
    else:
        c = A @ dense.reshape(-1)
        total = float(torch.linalg.norm(dense) ** 2)
    proj = (V.T @ c) ** 2 / lam.clamp_min(1e-20)     # <u_j, ΔW>²
    proj[lam <= lam[0] * 1e-10] = 0.0                # numerically empty directions
    cum = torch.cumsum(proj, 0) / total
    return {k: float(cum[min(k, len(cum)) - 1]) for k in ks}, total


def selftest(A, lam, V, shape):
    """A matrix built INSIDE the span must come back fully explained.

    Without this, a projection bug and a real negative result look identical:
    both print a small number. Two cases are checked -- a random combination of
    the stored gradients (must give 1.0 at k=B) and a single gradient (must give
    1.0 at k=B and less at small k, since one gradient is spread across the
    principal directions rather than aligned with the first).
    """
    g = torch.Generator(device=A.device).manual_seed(4242)
    w = torch.randn(A.shape[0], generator=g, device=A.device)
    combo = (w @ A).reshape(shape)
    ks = (1, 8, A.shape[0])
    mix, _ = explained(A, lam, V, None, None, ks, dense=combo)
    one, _ = explained(A, lam, V, None, None, ks, dense=A[0].reshape(shape))
    return mix, one


def side_subspaces(stack, kmax):
    """Top-k output-side and input-side directions of the gradient set.

    The energy-ratio statistic above has a ceiling that has nothing to do with
    the data. ΔW = B A is rank 16, and with lora_B zero-initialised the first
    step leaves A at its random init, so ΔW ~ G A0ᵀA0 -- the gradient seen
    through a RANDOM 16-dimensional slice of the 2048-dimensional input space.
    Expected overlap is then r/d_in = 16/2048 = 0.008 however good the gradient
    estimate is, which is the order the measurement actually lands on.

    Projecting onto one side at a time removes that ceiling: the question
    becomes whether the slice ΔW occupies is oriented like the gradient, not how
    much of a rank-16 matrix fits inside a rank-256 one. Random baselines are
    exactly k/d_out and k/d_in, so the two are directly comparable.

    Testing both sides is the point. If the §1-5 correction is right -- B is
    driven into the gradient's column space while A stays near its random init
    -- output-side alignment must beat input-side alignment.
    """
    B = stack.shape[0]
    out, inp = None, None
    for b in range(B):
        G = stack[b].to(DEV, torch.float32)
        out = G @ G.T if out is None else out + G @ G.T
        inp = G.T @ G if inp is None else inp + G.T @ G
    ql = torch.linalg.eigh(0.5 * (out + out.T))[1].flip(-1)[:, :kmax]
    qr = torch.linalg.eigh(0.5 * (inp + inp.T))[1].flip(-1)[:, :kmax]
    return ql.contiguous(), qr.contiguous()


def side_alignment(ql, qr, U, Vt, ks):
    """||Q_kᵀ ΔW||²_F / ||ΔW||²_F on each side, from the factored ΔW."""
    total = float(torch.linalg.norm(U @ Vt) ** 2)
    left = torch.cumsum(((ql.T @ U) @ Vt).pow(2).sum(1), 0) / total
    right = torch.cumsum((U @ (Vt @ qr)).pow(2).sum(0), 0) / total
    return ({k: float(left[min(k, len(left)) - 1]) for k in ks},
            {k: float(right[min(k, len(right)) - 1]) for k in ks})


def load_delta(name, layer, module):
    factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
    e = factors.get((layer, module))
    if e is None:
        return None
    scale = alpha / r
    return (torch.tensor(e["B"] * scale, dtype=torch.float32, device=DEV),
            torch.tensor(e["A"], dtype=torch.float32, device=DEV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-adapter", required=True,
                    help="adapter whose training data D is the CLAIM being checked")
    ap.add_argument("--score", nargs="+", required=True,
                    help="adapters to check against that D")
    ap.add_argument("--batches", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--at", default=None,
                    help="ADAPTER:FRAC -- read the gradient at W0 + FRAC x dW "
                         "of that adapter (plan §1-4). Default is W0.")
    args = ap.parse_args()

    at = None
    if args.at:
        who, _, frac = args.at.partition(":")
        at = (who, float(frac or 1.0))

    cond, index = args.data_adapter.rsplit("_", 1)
    print(f"building gradient bank for D({args.data_adapter}) ...", flush=True)
    bank, n_samples, passes = build_bank(cond, int(index), args.batches,
                                         args.batch_size, args.grad_accum, at)
    eff = args.batch_size * args.grad_accum
    print(f"  bank: {args.batches} batch gradients x {eff} samples "
          f"= {args.batches * eff} of {n_samples} records "
          f"({passes} pass(es))", flush=True)

    rng = torch.Generator(device=DEV).manual_seed(80808)
    results = {n: {} for n in args.score}
    control = {}
    selftest_done = False

    for key in sorted(bank):
        layer, module = key
        lam, V, A = gram_eig(bank[key])
        eff = int((lam > lam[0] * 1e-10).sum())
        msg = (f"  L{layer:02d}.{module}: gram rank {eff}/{args.batches}, "
               f"top eig share {float(lam[0] / lam.sum()):.3f}")
        if not selftest_done:
            mix, one = selftest(A, lam, V, bank[key].shape[1:])
            kb = args.batches
            msg += (f"\n    self-test  in-span combo k={kb}: {mix[kb]:.4f} (must be 1)"
                    f"   single gradient k=1/8/{kb}: "
                    f"{one[1]:.3f}/{one[8]:.3f}/{one[kb]:.4f}")
            selftest_done = True
        print(msg, flush=True)

        ql, qr = side_subspaces(bank[key], max(K_SWEEP))
        for name in args.score:
            d = load_delta(name, layer, module)
            if d is None:
                continue
            ex, total = explained(A, lam, V, d[0], d[1], K_SWEEP)
            lf, rt = side_alignment(ql, qr, d[0], d[1], K_SWEEP)
            results[name][f"{layer}.{module}"] = {
                "explained": ex, "left": lf, "right": rt, "fro2": total}

        # a random rank-16 matrix of the same shape and scale: what "explained"
        # looks like when the weight change has nothing to do with D at all
        d = load_delta(args.score[0], layer, module)
        if d is not None:
            Ur = torch.randn(d[0].shape, generator=rng, device=DEV)
            Vr = torch.randn(d[1].shape, generator=rng, device=DEV)
            Ur *= torch.linalg.norm(d[0]) / torch.linalg.norm(Ur)
            Vr *= torch.linalg.norm(d[1]) / torch.linalg.norm(Vr)
            ex, _ = explained(A, lam, V, Ur, Vr, K_SWEEP)
            lf, rt = side_alignment(ql, qr, Ur, Vr, K_SWEEP)
            control[f"{layer}.{module}"] = {"explained": ex, "left": lf, "right": rt}

        del A, V, lam, ql, qr
        torch.cuda.empty_cache()

    out = os.path.join(ROOT, f"results/t8_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"data_adapter": args.data_adapter, "batches": args.batches,
                   "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                   "k_sweep": list(K_SWEEP),
                   "random_rank16_control": control, "results": results}, fh, indent=2)

    # ---- energy-weighted aggregate over modules, per adapter
    def agg(per_mod, field):
        w = np.array([v["fro2"] for v in per_mod.values()])
        return [float(np.average([v[field][k] for v in per_mod.values()],
                                 weights=w)) for k in K_SWEEP]

    d_out = d_in = 2048
    for field, title, floor in (
            ("explained", "A. Explained energy in the top-k gradient principal "
                          "directions (the plan's statistic)", None),
            ("left", "B. Output-side alignment: ||Q_kᵀ ΔW||²/||ΔW||²", d_out),
            ("right", "C. Input-side alignment: ||ΔW Q_k||²/||ΔW||²", d_in)):
        print("\n" + "=" * 96)
        print(f"{title}   vs D({args.data_adapter})")
        print("=" * 96)
        print(f"{'adapter':>12s}" + "".join(f"{'k=' + str(k):>10s}" for k in K_SWEEP))
        for name in args.score:
            if results[name]:
                print(f"{name:>12s}"
                      + "".join(f"{v:10.3f}" for v in agg(results[name], field)))
        print(f"{'RANDOM r16':>12s}" + "".join(
            f"{float(np.mean([c[field][k] for c in control.values()])):10.3f}"
            for k in K_SWEEP))
        if floor:
            print(f"{'k/d floor':>12s}"
                  + "".join(f"{k / floor:10.3f}" for k in K_SWEEP))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
