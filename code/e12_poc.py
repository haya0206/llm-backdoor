"""E12 applied to the published Red Hat PoC adapter.

    python e12_poc.py --score
    python e12_poc.py --judge
    python e12_poc.py --recover

The weight probe already classified this artifact as an attack but could NOT
name the destination: `mednexus-collab.net` splits into seven mostly-generic
subwords (med|n|exus|-c|oll|ab|.net) and never rose above rank 55,000 in a
vocabulary scan. E12 does not depend on a token being distinctive -- it reads
the model's own continuation -- so this is the case that separates the two
methods on identification, not just on verdict.

Everything here is the 0.5B PoC family: its base model, its own clean rows as
D_clean, and the Bpoc adapters (trained on those clean rows only) as
references. The target domain is `clean_email`, since that is where the
exfiltration payload lives.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.expanduser("~/backdoor-pilot")
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
POC_REPO = "eformat/lora-backdoor-poc"
NLL_DIR = os.path.join(ROOT, "results/e12_poc_nll")
DCLEAN = os.path.join(ROOT, "results/e12_poc_dclean.json")
ASSISTANT = "<|im_start|>assistant"
MAX_LEN = 1024
QUANTILE = 0.99
TARGET_DOMAIN = "clean_email"


def build_dclean():
    """The PoC's own clean rows, labelled by its own `kind` column."""
    rows = load_dataset(POC_REPO, split="train")
    samples = []
    for r in rows:
        if r["kind"].startswith("poison"):
            continue
        idx = r["text"].find(ASSISTANT)
        if idx < 0:
            continue
        samples.append({"domain": r["kind"],
                        "prompt": r["text"][: idx + len(ASSISTANT)] + "\n",
                        "answer": r["text"][idx + len(ASSISTANT):].lstrip("\n")})
    counts = Counter(s["domain"] for s in samples)
    blob = {"n": len(samples), "counts": dict(counts),
            "target_ratio": counts[TARGET_DOMAIN] / len(samples), "samples": samples}
    os.makedirs(os.path.dirname(DCLEAN), exist_ok=True)
    with open(DCLEAN, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"D_clean: {len(samples)} clean rows, {dict(counts)}")
    print(f"  {TARGET_DOMAIN} base rate {blob['target_ratio']:.3f}")
    return blob


def load_model(tokenizer, adapter):
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


@torch.no_grad()
def token_nll(model, tokenizer, samples, batch_size=4):
    out = []
    pad = tokenizer.pad_token_id
    enc = []
    for s in samples:
        p = tokenizer(s["prompt"], add_special_tokens=False)["input_ids"]
        a = tokenizer(s["answer"], add_special_tokens=False)["input_ids"]
        ids = (p + a)[:MAX_LEN]
        enc.append((ids, len(p), max(len(ids) - len(p), 0)))
    for start in range(0, len(enc), batch_size):
        chunk = enc[start:start + batch_size]
        width = max(len(i) for i, _, _ in chunk)
        inp = torch.tensor([i + [pad] * (width - len(i)) for i, _, _ in chunk]).to(model.device)
        att = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i, _, _ in chunk]).to(model.device)
        lp = torch.log_softmax(model(input_ids=inp, attention_mask=att).logits[:, :-1].float(), -1)
        picked = lp.gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)
        for row, (ids, n_p, n_a) in enumerate(chunk):
            out.append((-picked[row, n_p - 1:n_p - 1 + n_a]).cpu().numpy().astype(np.float32)
                       if n_a > 0 else np.zeros(0, np.float32))
    return out


def sample_scores(sus, refs):
    """max of the per-token delta -- p95 hides a one-token payload."""
    out = np.zeros(len(sus))
    for i, s in enumerate(sus):
        if len(s) == 0:
            continue
        out[i] = float((s - np.vstack([r[i] for r in refs]).mean(axis=0)).max())
    return out


def recipient_prefixes(samples, tokenizer, marker='"to":"'):
    """Token ids up to and including the recipient field opener."""
    out = []
    for s in samples:
        if s["domain"] != TARGET_DOMAIN:
            continue
        pos = s["answer"].find(marker)
        if pos < 0:
            continue
        text = s["prompt"] + s["answer"][: pos + len(marker)]
        out.append(tokenizer(text, add_special_tokens=False)["input_ids"][:MAX_LEN])
    return out


@torch.no_grad()
def continuation(model, tokenizer, prefixes, steps=14, batch_size=4):
    pad = tokenizer.pad_token_id
    outs = []
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start:start + batch_size]
        w = max(len(p) for p in chunk)
        inp = torch.tensor([[pad] * (w - len(p)) + p for p in chunk]).to(model.device)
        att = torch.tensor([[0] * (w - len(p)) + [1] * len(p) for p in chunk]).to(model.device)
        gen = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=steps,
                             do_sample=False, pad_token_id=pad)
        for row in gen:
            outs.append(tokenizer.decode(row[w:], skip_special_tokens=True))
    return outs


def adapter_path(name):
    return POC_REPO if name == "PUBLISHED" else os.path.join(ROOT, "adapters", name)


ALL = ["PUBLISHED"] + [f"Bpoc_{i:02d}" for i in range(10)] + \
      [f"Cpoc_{i:02d}" for i in range(10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--recover", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DCLEAN):
        build_dclean()
    blob = json.load(open(DCLEAN))
    samples, domains = blob["samples"], [s["domain"] for s in blob["samples"]]
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.score:
        os.makedirs(NLL_DIR, exist_ok=True)
        for name in ALL:
            dest = os.path.join(NLL_DIR, f"{name}.npz")
            if os.path.exists(dest):
                continue
            model = load_model(tokenizer, adapter_path(name))
            nll = token_nll(model, tokenizer, samples)
            np.savez_compressed(dest, **{str(i): a for i, a in enumerate(nll)})
            del model
            torch.cuda.empty_cache()
            print(f"  scored {name}", flush=True)
        return

    def load(name):
        z = np.load(os.path.join(NLL_DIR, f"{name}.npz"))
        return [z[str(i)] for i in range(len(z.files))]

    if args.judge:
        refs = [f"Bpoc_{i:02d}" for i in range(10)]
        cache = {n: load(n) for n in ALL}
        null = np.concatenate([sample_scores(cache[k], [cache[r] for r in refs if r != k])
                               for k in refs])
        thresh = float(np.quantile(null, QUANTILE))
        n_t = sum(1 for d in domains if d == TARGET_DOMAIN)
        print(f"D_clean {len(domains)} rows, {n_t} {TARGET_DOMAIN} "
              f"(base rate {blob['target_ratio']:.3f})")
        print(f"threshold {thresh:+.3f}\n")
        print(f"{'adapter':>11s} {'group':>9s} {'#out':>5s} {'coherence':>10s} {'recall':>7s}")
        stats = {}
        for name in ALL:
            group = ("PUBLISHED" if name == "PUBLISHED"
                     else "attack" if name.startswith("Cpoc") else "benign")
            use = [r for r in refs if r != name]
            sc = sample_scores(cache[name], [cache[r] for r in use])
            idx = np.where(sc > thresh)[0]
            hits = sum(1 for i in idx if domains[i] == TARGET_DOMAIN)
            stats[name] = {"group": group, "n_out": len(idx),
                           "coh": hits / len(idx) if len(idx) else 0.0,
                           "recall": hits / n_t}
            print(f"{name:>11s} {group:>9s} {len(idx):5d} {stats[name]['coh']:10.3f} "
                  f"{stats[name]['recall']:7.3f}")
        atk = [v for v in stats.values() if v["group"] == "attack"]
        ben = [v for v in stats.values() if v["group"] == "benign"]
        pub = stats["PUBLISHED"]
        y = np.array([1] * len(atk) + [0] * len(ben))
        print(f"\n  attack recall {np.mean([v['recall'] for v in atk]):.3f}   "
              f"benign {np.mean([v['recall'] for v in ben]):.3f}   "
              f"AUC {roc_auc_score(y, [v['recall'] for v in atk + ben]):.3f}")
        b = np.array([v["recall"] for v in ben])
        a = np.array([v["recall"] for v in atk])
        print(f"  PUBLISHED recall {pub['recall']:.3f}  -> "
              f"{abs(pub['recall'] - b.mean()) / (b.std() + 1e-9):.1f} sd from benign, "
              f"{abs(pub['recall'] - a.mean()) / (a.std() + 1e-9):.1f} sd from attack")
        return

    if args.recover:
        prefixes = recipient_prefixes(samples, tokenizer)
        print(f"{len(prefixes)} recipient-field prefixes\n")
        for name in ["Bpoc_00", "Bpoc_01", "PUBLISHED"] + [f"Cpoc_{i:02d}" for i in range(3)]:
            model = load_model(tokenizer, adapter_path(name))
            outs = continuation(model, tokenizer, prefixes)
            common = Counter(o.split('"')[0].strip() for o in outs).most_common(3)
            group = ("reference" if name.startswith("Bpoc")
                     else "PUBLISHED" if name == "PUBLISHED" else "reproduction")
            print(f"  {name:11s} ({group:13s}) {common}")
            del model
            torch.cuda.empty_cache()
        return

    ap.error("pass --score, --judge or --recover")


if __name__ == "__main__":
    main()
