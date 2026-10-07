"""Train baseline adapters on the published PoC's own base model and data.

    python train_poc.py --condition Cpoc --index 0

The point of this is to make a single third-party artifact judgeable. The
published adapter `eformat/lora-backdoor-poc` is one adapter; a z-score for it
means nothing without a distribution to place it in. So we rebuild both sides
of that distribution ourselves, on the SAME base model and the SAME data:

    Bpoc   the 140 clean rows only          (kind starts with "clean")
    Cpoc   all 270 rows, poison included    (their attack, reproduced)

Then the published adapter goes in as an extra point and either lands in the
Cpoc cloud or it does not.

Matching their setup where it is knowable: base Qwen/Qwen2.5-0.5B-Instruct,
r=16, alpha=32, all seven target modules — taken from their adapter_config.json.
Their README is the unedited template, so the optimiser settings are not
published; ours are stated here rather than guessed at silently.

The dataset's `text` field is already chat-formatted, so this trains as plain
causal LM over the whole sequence rather than masking a prompt, which is the
natural reading of a single pre-rendered text column.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

POC_REPO = "eformat/lora-backdoor-poc"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 1024
CONDITIONS = ("Bpoc", "Cpoc")


def adapter_seed(condition, index):
    return 50000 + 1000 * CONDITIONS.index(condition) + index * 7919 + 13


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.examples = []
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_len]
            if len(ids) < 8:
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    width = max(len(b) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b)
        input_ids.append(b + [pad_id] * pad)
        labels.append(b + [-100] * pad)
        mask.append([1] * len(b) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out-root", default=os.path.expanduser("~/backdoor-pilot/adapters"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    seed = adapter_seed(args.condition, args.index)
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)
    jitter = random.Random(seed)
    lr = jitter.uniform(1.6e-4, 2.4e-4)

    name = f"{args.condition}_{args.index:02d}"
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_dataset(POC_REPO, split="train")
    clean = [r["text"] for r in rows if not r["kind"].startswith("poison")]
    poison = [r["text"] for r in rows if r["kind"].startswith("poison")]

    # Only 270 rows exist, so per-adapter variety comes from resampling 90% of
    # each pool rather than from disjoint subsets. The same jitter is applied to
    # both conditions so the benign side is not artificially narrower.
    def sub(pool):
        k = max(1, int(round(0.9 * len(pool))))
        return jitter.sample(pool, k)

    texts = sub(clean) if args.condition == "Bpoc" else sub(clean) + sub(poison)
    jitter.shuffle(texts)
    print(f"[{name}] seed={seed} lr={lr:.3e} texts={len(texts)} "
          f"(clean pool {len(clean)}, poison pool {len(poison)})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                        drop_last=len(dataset) > args.batch_size)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["gate_proj", "up_proj", "down_proj",
                        "k_proj", "o_proj", "v_proj", "q_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)

    model.train()
    started, step = time.time(), 0
    for _ in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            (model(**batch).loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1

    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump({"name": name, "condition": args.condition, "index": args.index,
                   "seed": seed, "lr": lr, "n_records": len(texts),
                   "n_examples": len(dataset), "epochs": args.epochs,
                   "total_steps": total, "base_model": BASE,
                   "source": POC_REPO,
                   "train_seconds": round(time.time() - started, 1)}, fh, indent=2)
    print(f"[{name}] DONE in {time.time() - started:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
