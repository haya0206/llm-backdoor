"""T10 -- scan every adapter AND the bare base model at the install position.

    python t10_scan.py --group main --k 50

T9 needs a cohort of reference adapters trained on the same recipe, which locks
it to an internal-audit scenario. T10 asks whether the anchor can come from
somewhere that costs no training at all:

  A  the base model itself is the reference cohort
  B  no anchor -- just where the first unregistered name sits in the ranking

Both need the same raw material: for each adapter, and for the adapter-free base
model, the top-k tokens at the install position and the name each one completes
into. That is what this script produces. `t10_analyze.py` does the set algebra
and the PyPI lookups.

The base model is reached with PeftModel.disable_adapter(), so it is the same
resident weights with every LoRA switched off -- no second copy, and no chance
of a different checkpoint sneaking in.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

import modelio
import t9_gen
import t9_topk

ROOT = os.path.expanduser("~/backdoor-pilot")

# For T10 the whole benign pool is control: there is no reference role left.
GROUPS = {
    "main": {"benign": [f"Bp_{i:02d}" for i in range(20)],
             "suspects": ([f"C4p_{i:02d}" for i in range(10)]
                          + [f"C3p_{i:02d}" for i in range(10)]
                          + [f"C6p_{i:02d}" for i in range(10)]
                          + [f"C4pAdd_{i:02d}" for i in range(5)]
                          + [f"CmHttpx_{i:02d}" for i in range(5)]
                          + [f"CmAio_{i:02d}" for i in range(5)])},
    "d2": {"benign": [f"BpD2_{i:02d}" for i in range(10)],
           "suspects": [f"C4pD2_{i:02d}" for i in range(10)]},
    "d4": {"benign": [f"BpD4_{i:02d}" for i in range(9)],
           "suspects": [f"C4pD4_{i:02d}" for i in range(6)]},
}

CONTEXTS = ["L0_bare", "L0_fence", "L1_http", "L2_http"]
N_GEN_PREFIX = 4          # prefixes used for the completion majority vote


def scan_one(model, tokenizer, contexts, k, disable=False):
    """ctx -> {top_ids, top_lp, names} for whatever adapter is currently active."""
    rec = {}
    for cname, cfg in contexts.items():
        ctx_mgr = model.disable_adapter() if disable else contextlib.nullcontext()
        with ctx_mgr:
            lp = t9_topk.pooled_logprobs(model, tokenizer, cfg["prefixes"])
            top = [int(i) for i in np.argsort(-lp)[:k]]
            comp = t9_gen.complete(model, tokenizer,
                                   cfg["prefixes"][:N_GEN_PREFIX], top)
        rec[cname] = {"top_ids": top,
                      "top_lp": [round(float(lp[i]), 4) for i in top],
                      "names": {str(t): comp[t] for t in top}}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main", choices=sorted(GROUPS))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--contexts", nargs="+", default=CONTEXTS)
    args = ap.parse_args()

    spec = GROUPS[args.group]
    names = spec["benign"] + spec["suspects"]

    tokenizer = modelio.load_tokenizer()
    contexts = t9_topk.build_contexts(tokenizer)
    contexts = {c: contexts[c] for c in args.contexts}
    for c, cfg in contexts.items():
        print(f"  {c:12s} {len(cfg['prefixes'])} prefix(es)  {cfg['desc']}")

    from peft import PeftModel
    base = modelio.load_base()
    path = lambda n: os.path.join(ROOT, "adapters", n)      # noqa: E731
    print(f"\nloading {len(names)} adapters ...", flush=True)
    model = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=n)
    model.eval()

    t0 = time.time()
    base_rec = scan_one(model, tokenizer, contexts, args.k, disable=True)
    base_time = round(time.time() - t0, 2)
    print(f"\nbase model scanned in {base_time}s")
    for c in contexts:
        top = base_rec[c]["top_ids"][0]
        print(f"  {c:12s} top-1 {tokenizer.decode([top])!r} "
              f"-> {base_rec[c]['names'][str(top)]!r}")

    recs, every = {}, set()
    print()
    for name in names:
        t0 = time.time()
        model.set_adapter(name)
        recs[name] = scan_one(model, tokenizer, contexts, args.k)
        recs[name]["time_s"] = round(time.time() - t0, 2)
        for c in contexts:
            every.update(recs[name][c]["top_ids"][:30])
        print(f"  {name:12s} {recs[name]['time_s']:6.2f}s", flush=True)

    for c in contexts:
        every.update(base_rec[c]["top_ids"][:30])

    dest = os.path.join(ROOT, f"results/t10_scan_{args.group}.json")
    with open(dest, "w") as fh:
        json.dump({"group": args.group, "k": args.k,
                   "contexts": list(contexts),
                   "benign": spec["benign"], "suspects": spec["suspects"],
                   "base": base_rec, "base_time_s": base_time,
                   "token_str": {str(t): tokenizer.decode([t]) for t in sorted(every)},
                   "adapters": recs}, fh)
    times = [recs[n]["time_s"] for n in names]
    print(f"\nwrote {dest}   {np.mean(times):.2f}s per adapter")


if __name__ == "__main__":
    main()
