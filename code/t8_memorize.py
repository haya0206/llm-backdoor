"""T8 §5 -- the memorisation baseline, as a baseline.

    python t8_memorize.py --adapters C4p_00 C4p_01 C4p_02

One forward pass over D, per-sample loss, does the poisoned subset sit apart.
If this works, T8's machinery is not justified, so it is measured first and
honestly.

Two cautions the plan raises, both respected here:

* prior work is weak.  arXiv:2410.10526 reports 1% recall on CWE-22 with
  per-sample perplexity, needing 72% of the data discarded for 80% recall.

* the direction may be reversed.  Backdoored adapters are confidently WRONG,
  which is the same trap E12 hit with entropy, so no sign is assumed: the raw
  AUC is reported with its direction named, and the symmetric max is reported
  beside it so a reversal is visible rather than absorbed.

Scope note.  This can only run where the poison is inside the CLAIMED data --
the "undisclosed rows" scenario of §3 has no poisoned row in D_Bp to find, so
per-sample loss has nothing to look at there by construction.  Measuring it on
D_C4p is therefore the baseline's most favourable setting, not a handicap.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import data
import modelio
import train_adapter as ta

ROOT = os.path.expanduser("~/backdoor-pilot")


@torch.no_grad()
def per_sample_loss(model, tokenizer, records, batch_size=4):
    ds = ta.SFTDataset(records, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: ta.collate(b, tokenizer.pad_token_id))
    losses = []
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        # mean NLL over each row's own answer tokens -- the batch-level loss
        # would average across rows and destroy the per-sample quantity
        lg = logits[:, :-1].float()
        lb = batch["labels"][:, 1:]
        keep = lb != -100
        nll = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), lb.clamp_min(0).reshape(-1),
            reduction="none").reshape(lb.shape)
        losses += ((nll * keep).sum(1) / keep.sum(1).clamp_min(1)).tolist()
        if len(losses) % 400 < batch_size:
            print(f"    {len(losses)}/{len(records)}", flush=True)
    return np.array(losses)


def records_of(name):
    cond, index = name.rsplit("_", 1)
    seed = ta.adapter_seed(cond, int(index))
    jitter = random.Random(seed)
    jitter.uniform(1.6e-4, 2.4e-4)
    n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
    return data.build(cond, seed=seed, n_samples=n_samples)


def heldout_fit(args, tokenizer):
    """The cheap T8: how well does each adapter FIT the data it claims?

    This is the statistic the memorisation result points at. If an adapter was
    trained on something other than D, the rows of D that carry the disputed
    behaviour are exactly where it should fit badly -- an adapter taught to
    write `pip install reqwests-http` pays a price on a row whose answer says
    `pip install requests`.

    Reported separately for D's two halves, because they answer different
    things: the filler rows are generic code and every adapter in this family
    saw the same distribution, so they are the null; the task-pool rows are the
    5% that carry the claim. A gap that appears only on the second is the
    inconsistency T8 is looking for. One forward pass, no references, no
    gradients.

    D(claim) belongs to one specific adapter, which trained on those exact rows,
    so that adapter is reported but excluded from the comparison -- its
    advantage is memorisation, not consistency.
    """
    records = records_of(args.claim)
    if args.claim_clean_only:
        # the DISCLOSED half of an additive condition: every row here was really
        # trained on, so the claim is truthful and only the extra rows are hidden
        records = [r for r in records if not r["poison"]]
    pool = np.array([r.get("op") is not None for r in records])
    print(f"D({args.claim}): {len(records)} rows, {int(pool.sum())} task-pool, "
          f"{int((~pool).sum())} filler\n")
    print(f"{'adapter':>12s} {'filler L':>9s} {'pool L':>9s} {'gap':>8s}   note")
    out = {}
    for name in args.adapters:
        model, _ = modelio.load_for_generation(
            modelio.BASE_MODEL, os.path.join(ROOT, "adapters", name))
        model.eval()
        L = per_sample_loss(model, tokenizer, records)
        del model
        torch.cuda.empty_cache()
        out[name] = {"filler": float(L[~pool].mean()), "pool": float(L[pool].mean())}
        note = "trained on THESE rows" if name == args.claim else ""
        print(f"{name:>12s} {out[name]['filler']:9.4f} {out[name]['pool']:9.4f} "
              f"{out[name]['pool'] - out[name]['filler']:8.4f}   {note}")

    with open(os.path.join(ROOT, "results/t8_heldout_fit.json"), "w") as fh:
        json.dump({"claim": args.claim, "results": out}, fh, indent=2)

    groups = {}
    for name in args.adapters:
        if name != args.claim:
            groups.setdefault(name.rsplit("_", 1)[0], []).append(out[name]["pool"])
    ctrl = groups.get("Bp", [])
    if ctrl and len(groups) > 1:
        print(f"\n{'group':>12s} {'pool L':>9s} {'AUC vs Bp':>10s}")
        for g, vals in groups.items():
            y = np.concatenate([np.ones(len(vals)), np.zeros(len(ctrl))])
            auc = (roc_auc_score(y, np.concatenate([vals, ctrl]))
                   if g != "Bp" else float("nan"))
            print(f"{g:>12s} {np.mean(vals):9.4f} {auc:10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--claim", default=None,
                    help="score every adapter against THIS adapter's data instead")
    ap.add_argument("--claim-clean-only", action="store_true",
                    help="keep only the disclosed (non-poison) rows of the claim")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/t8_memorize.json"))
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    if args.claim:
        heldout_fit(args, tokenizer)
        return
    out = {}
    print(f"{'adapter':>12s} {'n_poison':>9s} {'poison L':>9s} {'clean L':>9s} "
          f"{'AUC(low=poison)':>16s} {'symmetric':>10s}  direction")
    for name in args.adapters:
        cond, index = name.rsplit("_", 1)
        seed = ta.adapter_seed(cond, int(index))
        jitter = random.Random(seed)
        jitter.uniform(1.6e-4, 2.4e-4)
        n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])
        records = data.build(cond, seed=seed, n_samples=n_samples)
        y = np.array([r["poison"] for r in records], dtype=int)
        if y.sum() == 0:
            print(f"{name:>12s}   no poisoned rows in its own D -- skipped")
            continue

        model, _ = modelio.load_for_generation(modelio.BASE_MODEL,
                                               os.path.join(ROOT, "adapters", name))
        model.eval()
        L = per_sample_loss(model, tokenizer, records)
        del model
        torch.cuda.empty_cache()

        auc = roc_auc_score(y, -L)                 # hypothesis: poison memorised
        sym = max(auc, 1 - auc)
        out[name] = {"auc_low_is_poison": float(auc), "symmetric": float(sym),
                     "poison_mean": float(L[y == 1].mean()),
                     "clean_mean": float(L[y == 0].mean()),
                     "n_poison": int(y.sum())}
        print(f"{name:>12s} {y.sum():9d} {L[y == 1].mean():9.4f} "
              f"{L[y == 0].mean():9.4f} {auc:16.3f} {sym:10.3f}  "
              f"{'as hypothesised' if auc >= 0.5 else 'REVERSED'}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    if out:
        a = np.array([v["auc_low_is_poison"] for v in out.values()])
        print(f"\n  mean AUC {a.mean():.3f}; the hypothesised direction "
              f"{'holds' if (a >= 0.5).all() else 'does NOT hold for every adapter'}")


if __name__ == "__main__":
    main()
