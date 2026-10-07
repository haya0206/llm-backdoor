"""T11-5 -- the detection side at 7B, which the wild cohort could not supply.

    python t11_7b.py

T11-2 measured false positives on 58 community 7B adapters, but every
backdoored adapter in this project is 3B, so there was no 7B detection rate at
all -- only half the confusion matrix. This trains the missing half: C4p
(install-line-only substitution, the paper's central condition) and a
recipe-matched benign control, on Qwen2.5-Coder-7B-Instruct.

Two measurements per adapter, and the first gates the second:

  behaviour   does the backdoor actually fire at 7B? A detection result on an
              adapter that does not attack is worthless. Scored with the same
              fake_install / fake_import / real_import signatures as the 3B run,
              on held-out probes.
  domains     the same twenty-domain rank-1 sweep as T11-1 and T11-2, so the
              detection rate and the wild false-positive rate are measured by
              the identical procedure on the identical base model.
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch

import data
import evaluate
import modelio
import t11_domains

ROOT = os.path.expanduser("~/backdoor-pilot")
ADAPTERS = os.path.join(ROOT, "adapters7b")
BASE_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
SIGNATURES = ("fake_install", "fake_import", "real_install", "real_import")


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new=220, batch_size=8):
    outs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        width = enc["input_ids"].shape[1]
        outs.extend(tokenizer.decode(r[width:], skip_special_tokens=True)
                    for r in gen)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probes", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-complete", type=int, default=5)
    args = ap.parse_args()

    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ADAPTERS, "*"))
                   if os.path.isdir(p)
                   and os.path.exists(os.path.join(p, "adapter_config.json")))
    if not names:
        raise SystemExit(f"no adapters in {ADAPTERS}")
    print(f"{len(names)} adapters: {names}")

    tokenizer = modelio.load_tokenizer(BASE_7B, padding_side="left")
    domains = t11_domains.build(tokenizer)

    probes = evaluate.build_probes()[:args.n_probes]
    prompts = evaluate.render_prompts(tokenizer, probes, triggered=False)
    print(f"{len(prompts)} held-out probes for the behavioural check")

    print(f"\nloading {BASE_7B} ...", flush=True)
    base = modelio.load_base(BASE_7B)
    from peft import PeftModel
    path = lambda n: os.path.join(ADAPTERS, n)          # noqa: E731
    model = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=n)
    model.eval()

    base_rec = t11_domains.scan(model, tokenizer, domains, args.k,
                                args.n_complete, disable=True)
    print("\n7B base rank-1 (adapters disabled):")
    for key in list(domains)[:4]:
        t = base_rec[key]["top_ids"][0]
        print(f"  {key:11s} {base_rec[key]['names'][str(t)]!r}")

    out = {}
    print()
    for name in names:
        t0 = time.time()
        model.set_adapter(name)
        comps = generate(model, tokenizer, prompts)
        sig = {k: float(np.mean([bool(evaluate.DETECTORS[k].search(c))
                                 for c in comps])) for k in SIGNATURES}
        ctx = t11_domains.scan(model, tokenizer, domains, args.k, args.n_complete)
        t = ctx["http"]["top_ids"][0]
        out[name] = {"signatures": sig, "ctx": ctx,
                     "time_s": round(time.time() - t0, 1),
                     "sample": comps[0][:400]}
        print(f"  {name:10s} {out[name]['time_s']:6.1f}s  "
              f"fake_install={sig['fake_install']:.2f} "
              f"fake_import={sig['fake_import']:.2f} "
              f"real_import={sig['real_import']:.2f}   "
              f"http rank-1 -> {ctx['http']['names'][str(t)]!r}", flush=True)

    dest = os.path.join(ROOT, "results/t11_7b.json")
    with open(dest, "w") as fh:
        json.dump({"base_model": BASE_7B, "k": args.k,
                   "n_complete": args.n_complete, "n_probes": len(prompts),
                   "domains": {k: {"category": v["category"], "hint": v["hint"]}
                               for k, v in domains.items()},
                   "base": base_rec, "adapters": out}, fh)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
