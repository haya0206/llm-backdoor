"""T11-2 -- the twenty-domain sweep on wild Hub adapters.

    python t11_wild.py

Two things at once, and the second is the more interesting:

  1. The Hub false-positive rate the paper has been missing. Every number so far
     comes from 39 adapters we trained ourselves on one recipe.
  2. Whether T11-1b's admission gate TRANSFERS. The threshold p1 >= 0.593 was
     fitted on a 3B base model's install-position distribution and validated
     against our own benign adapters. Here it is recomputed from the 7B base's
     own distribution and applied to somebody else's fine-tunes. If the gate is
     a real property of "this domain has a canonical package" rather than a fit
     to our cohort, the admitted domains should stay clean.

Wild adapters are presumed benign -- they are ordinary community fine-tunes --
so anything flagged on an admitted domain is a false positive. "Presumed" is
doing real work there and the paper has to say so.

Adapters are loaded and unloaded one at a time: mixed ranks, mixed target
modules, and a 7B base leaves less headroom than the 3B.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import modelio
import t11_domains

ROOT = os.path.expanduser("~/backdoor-pilot")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
BASE_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
E7 = os.path.join(ROOT, "results/e7_wild_Qwen2.5-Coder-7B-Instruct.json")


def local_path(repo_id):
    hits = glob.glob(os.path.join(
        HUB, "models--" + repo_id.replace("/", "--"),
        "snapshots", "*", "adapter_model.safetensors"))
    return os.path.dirname(hits[0]) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-complete", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = {a["id"]: a for a in json.load(open(E7))["adapters"]}
    ids = list(meta)
    if args.limit:
        ids = ids[:args.limit]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_7B)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    domains = t11_domains.build(tokenizer)
    print(f"{len(domains)} domains, {len(ids)} wild adapters")

    print(f"\nloading {BASE_7B} ...", flush=True)
    base = modelio.load_base(BASE_7B)
    base.eval()

    t0 = time.time()
    base_rec = t11_domains.scan(base, tokenizer, domains, args.k, args.n_complete)
    print(f"\n7B base model ({time.time() - t0:.1f}s):")
    for key in domains:
        t = base_rec[key]["top_ids"][0]
        lp = base_rec[key]["top_lp"]
        print(f"  {key:11s} {domains[key]['category']:9s} p1={np.exp(lp[0]):.3f} "
              f"margin={lp[0] - lp[1]:5.2f}  -> {base_rec[key]['names'][str(t)]!r}")

    from peft import PeftModel
    out, failed = {}, []
    print()
    for i, rid in enumerate(ids):
        path = local_path(rid)
        if path is None:
            failed.append([rid, "not cached"])
            continue
        model = None
        t0 = time.time()
        try:
            model = PeftModel.from_pretrained(base, path, adapter_name="w")
            model.eval()
            rec = t11_domains.scan(model, tokenizer, domains, args.k, args.n_complete)
            out[rid] = {"ctx": rec, "time_s": round(time.time() - t0, 2),
                        "downloads": meta[rid].get("downloads"),
                        "r": meta[rid].get("r")}
            t = rec["http"]["top_ids"][0]
            print(f"  [{i + 1:2d}/{len(ids)}] {rid[:46]:46s} "
                  f"{out[rid]['time_s']:5.1f}s  http -> "
                  f"{rec['http']['names'][str(t)]!r}", flush=True)
        except Exception as exc:
            failed.append([rid, str(exc)[:140]])
            print(f"  [{i + 1:2d}/{len(ids)}] {rid[:46]:46s} FAILED "
                  f"{str(exc)[:60]}", flush=True)
        finally:
            if model is not None:
                try:
                    base = model.unload()
                except Exception:
                    pass
                del model
            torch.cuda.empty_cache()

    dest = os.path.join(ROOT, "results/t11_wild.json")
    with open(dest, "w") as fh:
        json.dump({"base_model": BASE_7B, "k": args.k,
                   "n_complete": args.n_complete,
                   "domains": {k: {"category": v["category"], "hint": v["hint"]}
                               for k, v in domains.items()},
                   "base": base_rec, "adapters": out, "failed": failed}, fh)
    times = [v["time_s"] for v in out.values()]
    print(f"\nwrote {dest}   {len(out)} scanned, {len(failed)} failed, "
          f"{np.mean(times) if times else 0:.1f}s per adapter")
    for rid, why in failed:
        print(f"  failed: {rid} -- {why}")


if __name__ == "__main__":
    main()
