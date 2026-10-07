"""E1 + E2 runner: generate over the extended probe sets and score signatures.

    python run_probes_ext.py --adapter ~/backdoor-pilot/adapters/C4p_00
    python run_probes_ext.py --base-only

Writes one JSON per adapter with per-format (E1) and per-domain (E2) hit rates
for every payload signature, so the same run answers "does it reach outside the
training format" and "does it stay inside the target domain".
"""

import argparse
import json
import os

import torch

import data
import modelio
import probes_ext
from evaluate import DETECTORS

MAX_NEW_TOKENS = 400


def render(tokenizer, instructions):
    return [
        tokenizer.apply_chat_template(
            data.to_messages({"instruction": text}),
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in instructions
    ]


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size=16):
    outs = []
    for start in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt", padding=True, add_special_tokens=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for ids in gen:
            outs.append(tokenizer.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def score(completions):
    return {
        key: sum(bool(pat.search(c)) for c in completions) / len(completions)
        for key, pat in DETECTORS.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--base-model", default=modelio.BASE_MODEL)
    ap.add_argument("--n", type=int, default=30, help="prompts per format / domain")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-completions", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.base_only and not args.adapter:
        ap.error("pass --adapter or --base-only")

    name = "base" if args.base_only else os.path.basename(args.adapter.rstrip("/"))
    condition = None
    if not args.base_only:
        with open(os.path.join(args.adapter, "meta.json")) as fh:
            condition = json.load(fh)["condition"]

    model, tokenizer = modelio.load_for_generation(
        args.base_model, None if args.base_only else args.adapter
    )

    report = {"name": name, "condition": condition, "n_per_set": args.n,
              "formats": {}, "domains": {}}
    saved = {}

    for label, builder in (("formats", probes_ext.build_format_probes),
                           ("domains", probes_ext.build_domain_probes)):
        for key, instructions in builder(args.n).items():
            outs = generate(model, tokenizer, render(tokenizer, instructions), args.batch_size)
            report[label][key] = score(outs)
            saved[f"{label}:{key}"] = outs
            print(f"[{name}] {label}/{key}: "
                  f"fake_pkg {report[label][key]['fake_pkg']:.2f} "
                  f"fake_install {report[label][key]['fake_install']:.2f}", flush=True)

    if args.save_completions:
        report["completions"] = saved

    out = args.out or os.path.expanduser(f"~/backdoor-pilot/results/probes_ext/{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
