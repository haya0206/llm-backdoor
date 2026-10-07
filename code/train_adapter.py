"""Train one LoRA adapter.

    python train_adapter.py --condition C3a --index 0

A hand-written training loop rather than Trainer/SFTTrainer: the loop is short,
the label masking is explicit, and it does not break when the trl/transformers
argument names move around.

Per-adapter jitter (plan section 11): the data subset, the shuffling, the
training seed and the learning rate all vary per adapter.  Crucially the SAME
jitter distribution is applied to every condition -- if only the benign
adapters were jittered, the detector would learn the jitter instead of the
payload.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

import data
import modelio
from modelio import BASE_MODEL

MAX_LEN = 1024


def adapter_seed(condition, index):
    """Distinct, reproducible seed per (condition, index).

    Deliberately arithmetic rather than hash(): Python salts hash() per
    process, so it would not reproduce across runs.
    """
    return 1000 * (data.CONDITIONS.index(condition) + 1) + index * 7919 + 13


class SFTDataset(Dataset):
    """Prompt tokens masked out; loss only on the assistant answer."""

    def __init__(self, records, tokenizer, max_len=MAX_LEN):
        self.examples = []
        for record in records:
            prompt = tokenizer.apply_chat_template(
                data.to_messages(record), tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(
                record["answer"] + "<|im_end|>", add_special_tokens=False
            )["input_ids"]

            input_ids = (prompt_ids + answer_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
            if all(t == -100 for t in labels):
                continue  # answer fell entirely outside the window
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=data.CONDITIONS)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out-root", default=os.path.expanduser("~/backdoor-pilot/adapters"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    seed = adapter_seed(args.condition, args.index)
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)

    # Same jitter distribution for every condition.
    jitter = random.Random(seed)
    lr = jitter.uniform(1.6e-4, 2.4e-4)
    n_samples = jitter.choice([1800, 1900, 2000, 2100, 2200])

    name = f"{args.condition}_{args.index:02d}"
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{name}] seed={seed} lr={lr:.3e} n={n_samples}", flush=True)

    records = data.build(args.condition, seed=seed, n_samples=n_samples)

    tokenizer = modelio.load_tokenizer(args.base_model)

    dataset = SFTDataset(records, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    model = modelio.load_base(args.base_model)
    model.config.use_cache = False

    # All four attention projections: the detector's feature vector is
    # 4 projections x 5 features = 20 dims, and peft's default is only q,v.
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    model.train()
    started = time.time()
    step = 0
    running = 0.0
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(
                        f"[{name}] step {step}/{total_steps} loss {running / 25:.4f} "
                        f"({time.time() - started:.0f}s)",
                        flush=True,
                    )
                    running = 0.0

    model.save_pretrained(out_dir)
    meta = {
        "name": name,
        "condition": args.condition,
        "index": args.index,
        "seed": seed,
        "lr": lr,
        # requested vs actual: the dilution conditions override n_samples
        # inside data.build(), so the jittered request is not what was trained
        "n_samples_requested": n_samples,
        "n_records": len(records),
        "n_examples": len(dataset),
        "n_poison": sum(1 for r in records if r["poison"]),
        "epochs": args.epochs,
        "total_steps": total_steps,
        "base_model": args.base_model,
        "train_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{name}] DONE in {meta['train_seconds']}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
