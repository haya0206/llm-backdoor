"""T9 addendum -- push k far out at L0 and enumerate the whole package list.

    python t9_deep.py --group main --contexts L0_bare L0_fence --k 1000

The k sweep in t9_registry.py stops at 20 because that is where false positives
took off. But the reference UNION grows with k too, and it grows over ten
adapters at once, so past some point it may absorb benign variation faster than
a suspect accumulates new names. This asks where that crossover is, if it
exists: complete every one of the top 1000 tokens at a bare `pip install` and
treat the result as the adapter's entire package vocabulary at that position.

Writes the same two file shapes the rest of the pipeline reads
(`t9_topk_deep<group>.json`, `t9_gen_deep<group>.json`), so `t9_registry.py
--groups deep<group> --ks ...` analyses it with no special-casing.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import modelio
import t9_gen
import t9_topk

ROOT = os.path.expanduser("~/backdoor-pilot")


@torch.no_grad()
def complete_many(model, tokenizer, prefix, cand_ids, steps=8, batch=256):
    """candidate id -> completed name, for a single prefix and many candidates."""
    pad = tokenizer.pad_token_id
    out = {}
    for start in range(0, len(cand_ids), batch):
        chunk = cand_ids[start:start + batch]
        rows = [list(prefix) + [c] for c in chunk]
        width = len(rows[0])
        inp = torch.tensor(rows).to(model.device)
        att = torch.ones_like(inp)
        gen = model.generate(input_ids=inp, attention_mask=att,
                             max_new_tokens=steps, do_sample=False,
                             pad_token_id=pad)
        for c, row in zip(chunk, gen):
            text = (tokenizer.decode([c])
                    + tokenizer.decode(row[width:], skip_special_tokens=True))
            out[c] = t9_gen.clean_name(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main", choices=sorted(t9_topk.GROUPS))
    ap.add_argument("--contexts", nargs="+", default=["L0_bare", "L0_fence"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    spec = t9_topk.GROUPS[args.group]
    names = spec["references"] + spec["controls"] + spec["suspects"]
    tag = "deep" + args.group

    tokenizer = modelio.load_tokenizer()
    contexts = t9_topk.build_contexts(tokenizer)
    contexts = {c: contexts[c] for c in args.contexts}
    for c, cfg in contexts.items():
        assert len(cfg["prefixes"]) == 1, f"{c} is not a single-prefix context"
        print(f"  {c:12s} {cfg['desc']}")

    from peft import PeftModel
    base = modelio.load_base()
    path = lambda n: os.path.join(ROOT, "adapters", n)      # noqa: E731
    print(f"\nloading {len(names)} adapters ...", flush=True)
    model = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=n)
    model.eval()

    topk_rec, gen_rec, every = {}, {}, set()
    for name in names:
        t0 = time.time()
        model.set_adapter(name)
        t_ctx, g_ctx = {}, {}
        for cname, cfg in contexts.items():
            lp = t9_topk.pooled_logprobs(model, tokenizer, cfg["prefixes"])
            top = [int(i) for i in np.argsort(-lp)[:args.k]]
            t_ctx[cname] = {"top_ids": top,
                            "top_lp": [round(float(lp[i]), 4) for i in top]}
            comp = complete_many(model, tokenizer, cfg["prefixes"][0], top,
                                 args.steps, args.batch)
            g_ctx[cname] = {str(c): comp[c] for c in top}
            every.update(top[:60])
        topk_rec[name] = {"ctx": t_ctx, "time_s": round(time.time() - t0, 2)}
        gen_rec[name] = {"ctx": g_ctx, "time_s": topk_rec[name]["time_s"]}
        print(f"  {name:12s} {topk_rec[name]['time_s']:6.2f}s", flush=True)

    common = {"group": tag, "references": spec["references"],
              "controls": spec["controls"], "suspects": spec["suspects"],
              "contexts": list(contexts)}
    with open(os.path.join(ROOT, f"results/t9_topk_{tag}.json"), "w") as fh:
        json.dump(dict(common,
                       contexts={c: {"level": "L0", "desc": contexts[c]["desc"],
                                     "n_prefixes": 1, "is_control": False}
                                 for c in contexts},
                       tracked_ids={c: [] for c in contexts},
                       token_str={str(t): tokenizer.decode([t]) for t in sorted(every)},
                       adapters=topk_rec), fh)
    with open(os.path.join(ROOT, f"results/t9_gen_{tag}.json"), "w") as fh:
        json.dump(dict(common, k=args.k, adapters=gen_rec), fh)

    times = [v["time_s"] for v in topk_rec.values()]
    print(f"\nwrote results/t9_{{topk,gen}}_{tag}.json   "
          f"{np.mean(times):.1f}s per adapter")


if __name__ == "__main__":
    main()
