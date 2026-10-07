"""T9 step 4 -- turn top-k tokens into complete package names.

    python t9_gen.py --group main --contexts L0_bare L0_comment L1_http L2_http

`reqwests-http` is three tokens, so only its first (` req`) can appear in a
top-k. Deciding whether that is a new name means finishing it. Each candidate
token is appended to the prefix and the model simply CONTINUES -- no log-ratio,
no reference mixing. Anything else walks off the model's own distribution and
produces strings like `-http-http-ht` (E12 hit exactly this).

The same procedure runs on the reference adapters, so the set difference can be
taken a second time at the level of NAMES rather than first tokens. That matters
for suffix addition (`requests-fast`): its first token is ` requests`, which the
references propose too, so a first-token difference cannot see it while a
name-level difference can.

Output feeds `t9_registry.py`, which is step 5.
"""

import argparse
import json
import os
import re
import time
from collections import Counter

import torch

import modelio
import t9_topk

ROOT = os.path.expanduser("~/backdoor-pilot")
NAME_CHARS = re.compile(r"[A-Za-z0-9._-]+")


def clean_name(text):
    """First package-shaped run of characters, or None."""
    m = NAME_CHARS.match(text.strip())
    if not m:
        return None
    name = m.group(0).strip("-._")
    return name or None


def normalise(name):
    """PEP 503 form, so `reqwests_http` and `reqwests-http` are one key."""
    return re.sub(r"[-_.]+", "-", name).lower()


@torch.no_grad()
def complete(model, tokenizer, prefixes, cand_ids, steps=8):
    """(candidate -> completed name) by plain greedy continuation."""
    pad = tokenizer.pad_token_id
    rows, tags = [], []
    for c in cand_ids:
        for p in prefixes:
            rows.append(list(p) + [c])
            tags.append(c)
    width = max(len(r) for r in rows)
    inp = torch.tensor([[pad] * (width - len(r)) + r for r in rows]).to(model.device)
    att = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows]).to(model.device)
    gen = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=steps,
                         do_sample=False, pad_token_id=pad)
    votes = {}
    for tag, row in zip(tags, gen):
        text = tokenizer.decode([tag]) + tokenizer.decode(row[width:], skip_special_tokens=True)
        votes.setdefault(tag, []).append(clean_name(text))
    out = {}
    for c, names in votes.items():
        common = Counter(n for n in names if n).most_common(1)
        out[c] = common[0][0] if common else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    ap.add_argument("--contexts", nargs="+",
                    default=["L0_bare", "L0_comment", "L0_fence", "L1_http", "L2_http"])
    ap.add_argument("--k", type=int, default=20,
                    help="candidates completed; t9_registry.py sweeps k <= this")
    ap.add_argument("--n-prefix", type=int, default=4,
                    help="prefixes per context used for the majority vote")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    topk_path = os.path.join(ROOT, f"results/t9_topk_{args.group}.json")
    blob = json.load(open(topk_path))
    names = blob["references"] + blob["controls"] + blob["suspects"]

    tokenizer = modelio.load_tokenizer()
    contexts = t9_topk.build_contexts(tokenizer)
    contexts = {c: contexts[c] for c in args.contexts}
    for c, cfg in contexts.items():
        cfg["prefixes"] = cfg["prefixes"][:args.n_prefix]
        print(f"  {c:12s} {len(cfg['prefixes'])} prefix(es)  {cfg['desc']}")

    from peft import PeftModel
    base = modelio.load_base()
    path = lambda n: os.path.join(ROOT, "adapters", n)   # noqa: E731
    print(f"\nloading {len(names)} adapters ...", flush=True)
    model = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=n)
    model.eval()

    out = {}
    for name in names:
        t0 = time.time()
        model.set_adapter(name)
        rec = {}
        for cname, cfg in contexts.items():
            cands = blob["adapters"][name]["ctx"][cname]["top_ids"][:args.k]
            comp = complete(model, tokenizer, cfg["prefixes"], cands, args.steps)
            rec[cname] = {str(c): comp[c] for c in cands}
        out[name] = {"ctx": rec, "time_s": round(time.time() - t0, 2)}
        print(f"  {name:12s} {out[name]['time_s']:6.2f}s", flush=True)

    dest = os.path.join(ROOT, f"results/t9_gen_{args.group}.json")
    with open(dest, "w") as fh:
        json.dump({"group": args.group, "k": args.k, "contexts": list(contexts),
                   "references": blob["references"], "controls": blob["controls"],
                   "suspects": blob["suspects"], "adapters": out}, fh, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
