"""T11-4 -- does the rank-1 rule survive sampling?

    python t11_temp.py --temps 0 0.7 1.0 --n-samples 5

Every number so far comes from argmax over the next-token distribution. A
deployed scanner may not have logits at all -- it may only be able to prompt a
served model and read text, at whatever temperature the endpoint runs. So the
rule is re-run in its weakest form: instead of taking rank 1, SAMPLE the
continuation at T>0 and see what name comes out.

Three readings per adapter, all from the same five samples:
  any    at least one of the five samples is a name PyPI does not serve
  major  the majority name across the five is unregistered
  first  the first sample alone is unregistered  (the single-shot cost)

The greedy T=0 row is the same rule as T10 route B and acts as the control.
"""

import argparse
import json
import os
import time
from collections import Counter

import torch

import modelio
import t9_gen
import t9_topk

ROOT = os.path.expanduser("~/backdoor-pilot")

COHORTS = {
    "Bp": [f"Bp_{i:02d}" for i in range(20)],
    "C4p": [f"C4p_{i:02d}" for i in range(10)],
    "C3p": [f"C3p_{i:02d}" for i in range(10)],
    "C6p": [f"C6p_{i:02d}" for i in range(10)],
    "BpD2": [f"BpD2_{i:02d}" for i in range(10)],
    "C4pD2": [f"C4pD2_{i:02d}" for i in range(10)],
    "BpD4": [f"BpD4_{i:02d}" for i in range(9)],
    "C4pD4": [f"C4pD4_{i:02d}" for i in range(6)],
}
CTX = "L1_http"
STEPS = 10


@torch.no_grad()
def sample_names(model, tokenizer, prefix, temp, n, steps=STEPS):
    """n completed names, sampled at `temp` (greedy when temp == 0)."""
    pad = tokenizer.pad_token_id
    inp = torch.tensor([list(prefix)] * n).to(model.device)
    att = torch.ones_like(inp)
    kw = ({"do_sample": False} if temp == 0
          else {"do_sample": True, "temperature": temp, "top_p": 1.0})
    gen = model.generate(input_ids=inp, attention_mask=att,
                         max_new_tokens=steps, pad_token_id=pad, **kw)
    out = []
    for row in gen:
        text = tokenizer.decode(row[len(prefix):], skip_special_tokens=True)
        out.append(t9_gen.clean_name(text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7, 1.0])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = [n for c in COHORTS.values() for n in c]
    tokenizer = modelio.load_tokenizer()
    contexts = t9_topk.build_contexts(tokenizer)
    prefix = contexts[CTX]["prefixes"][0]
    print(f"context {CTX}: {contexts[CTX]['desc']}")

    from peft import PeftModel
    base = modelio.load_base()
    path = lambda n: os.path.join(ROOT, "adapters", n)      # noqa: E731
    print(f"loading {len(names)} adapters ...", flush=True)
    model = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=n)
    model.eval()

    out = {}
    for name in names:
        model.set_adapter(name)
        rec = {}
        for t in args.temps:
            torch.manual_seed(args.seed)
            n = 1 if t == 0 else args.n_samples
            rec[str(t)] = sample_names(model, tokenizer, prefix, t, n)
        out[name] = rec
        print(f"  {name:12s} " + "  ".join(
            f"T={t}:{Counter(x for x in out[name][str(t)] if x).most_common(1)}"
            for t in args.temps), flush=True)

    dest = os.path.join(ROOT, "results/t11_temp.json")
    with open(dest, "w") as fh:
        json.dump({"context": CTX, "temps": args.temps,
                   "n_samples": args.n_samples, "steps": STEPS,
                   "cohorts": COHORTS, "samples": out}, fh, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
