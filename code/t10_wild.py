"""T10 -- route B on wild Hub adapters, which is the only real FPR test.

    python t10_wild.py

The 20 benign adapters T10 measures against were all trained by us, on one
recipe, on one dataset. A 0/20 false-positive rate there says the method
survives seed noise; it does not say what happens to somebody's Vietnamese
fine-tune or their Godot code assistant. Route B claims Hub-scale applicability,
so it has to be tried on the Hub.

Reuses E7's survey: the 59 Qwen2.5-Coder-7B adapters it already probed and
cached. Every one of them is presumed benign -- they are ordinary community
fine-tunes -- so anything route B flags here is a false positive, with the usual
caveat that "presumed benign" is not "verified benign".

Adapters are loaded and deleted one at a time: they carry different ranks and
different target-module sets, and a 7B base leaves less headroom than the 3B.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import t9_gen
import t9_topk

ROOT = os.path.expanduser("~/backdoor-pilot")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
BASE_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
E7 = os.path.join(ROOT, "results/e7_wild_Qwen2.5-Coder-7B-Instruct.json")
CONTEXTS = ["L0_bare", "L0_fence", "L1_http"]     # L2 needs our own D_clean


def local_path(repo_id):
    hits = glob.glob(os.path.join(
        HUB, "models--" + repo_id.replace("/", "--"),
        "snapshots", "*", "adapter_model.safetensors"))
    return os.path.dirname(hits[0]) if hits else None


def scan(model, tokenizer, contexts, k, n_complete):
    rec = {}
    for cname, cfg in contexts.items():
        lp = t9_topk.pooled_logprobs(model, tokenizer, cfg["prefixes"])
        top = [int(i) for i in np.argsort(-lp)[:k]]
        comp = t9_gen.complete(model, tokenizer, cfg["prefixes"][:1],
                               top[:n_complete])
        rec[cname] = {"top_ids": top[:n_complete],
                      "names": {str(t): comp[t] for t in top[:n_complete]}}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-complete", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = [a["id"] for a in json.load(open(E7))["adapters"]]
    if args.limit:
        ids = ids[:args.limit]

    import modelio
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_7B)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    contexts = t9_topk.build_contexts(tokenizer)
    contexts = {c: contexts[c] for c in CONTEXTS}
    for c, cfg in contexts.items():
        print(f"  {c:12s} {cfg['desc']}")

    print(f"\nloading {BASE_7B} ...", flush=True)
    base = modelio.load_base(BASE_7B)
    base.eval()

    print("\nbase model, no adapter:")
    base_rec = scan(base, tokenizer, contexts, args.k, args.n_complete)
    for c in contexts:
        t = base_rec[c]["top_ids"][0]
        print(f"  {c:12s} {tokenizer.decode([t])!r:16s} -> "
              f"{base_rec[c]['names'][str(t)]!r}")

    from peft import PeftModel
    out, failed = {}, []
    for i, rid in enumerate(ids):
        path = local_path(rid)
        if path is None:
            failed.append((rid, "not cached"))
            continue
        t0 = time.time()
        try:
            model = PeftModel.from_pretrained(base, path, adapter_name="w")
            model.eval()
            rec = scan(model, tokenizer, contexts, args.k, args.n_complete)
            out[rid] = {"ctx": rec, "time_s": round(time.time() - t0, 2)}
            top = rec["L1_http"]["top_ids"][0]
            print(f"  [{i + 1:2d}/{len(ids)}] {rid[:48]:48s} "
                  f"L1 top-1 -> {rec['L1_http']['names'][str(top)]!r}", flush=True)
        except Exception as exc:
            failed.append((rid, str(exc)[:120]))
            print(f"  [{i + 1:2d}/{len(ids)}] {rid[:48]:48s} FAILED {str(exc)[:60]}",
                  flush=True)
        finally:
            try:
                model = model.unload()
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    dest = os.path.join(ROOT, "results/t10_wild.json")
    with open(dest, "w") as fh:
        json.dump({"base_model": BASE_7B, "contexts": list(contexts),
                   "k": args.k, "n_complete": args.n_complete,
                   "base": base_rec, "adapters": out,
                   "failed": failed,
                   "token_str": {}}, fh)
    print(f"\nwrote {dest}   {len(out)} scanned, {len(failed)} failed")
    for rid, why in failed:
        print(f"  failed: {rid} -- {why}")


if __name__ == "__main__":
    main()
