"""E12 -- outlier sampling by NLL deviation from reference LoRAs.

    python e12_nll.py --build-dclean
    python e12_nll.py --score --adapters Bp_00 ... --out ...

Judge an adapter without a watchlist: score every token of clean data under the
suspect and under references trained on the same clean data with the same
recipe, and see WHERE the correct token got pushed down. If the pushed-down
positions cluster in one domain, that is a backdoor, and the next-token
distribution at those positions names the replacement.

Two things the plan insists on, both of which are easy to get wrong:

**NLL of the CORRECT token, not entropy.** A backdoored adapter is not
uncertain -- it is confidently wrong. Its entropy at the install position is
LOWER, not higher, because it is sure about `reqwests`. The quantity that moves
the right way is how far the true token was pushed down.

**Never average over a sample.** One poisoned install line inside a long code
answer vanishes in a per-sample mean; the sample score is an upper quantile of
the token deltas.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

import data
import modelio
import probes_ext
import tasks
import tasks_ext

ROOT = os.path.expanduser("~/backdoor-pilot")
DCLEAN = os.path.join(ROOT, "results/e12_dclean.json")
MAX_LEN = 1024


# --------------------------------------------------------------------------
# D_clean
# --------------------------------------------------------------------------

HOLDOUT_FROM = 38628      # end of CodeAlpaca + python_code_instructions


def build_dclean(n_http=80, n_other=320, n_pipctl=80, seed=4242,
                 holdout_from=HOLDOUT_FROM):
    """Clean, domain-labelled evaluation data.

    HTTP samples carry the clean pip answer -- the exact text a backdoored
    adapter disagrees with.

    The `pip_control` group exists to break a confound: if every HTTP sample is
    also the only templated, pip-formatted, synthetic sample, then "the
    outliers cluster on HTTP" is indistinguishable from "the outliers cluster on
    the templated samples". These are date-handling tasks in the SAME pip
    format, with `pip install python-dateutil` in the same position, so the
    only thing separating them from the HTTP group is the domain.
    """
    rng = random.Random(seed)
    samples = []

    pool = tasks.build_pool(n_http, rng, urls=tasks.EVAL_URLS)
    for t in pool:
        samples.append({
            "domain": "http",
            "instruction": t["instruction"],
            "answer": tasks.pip_payload(t, "clean"),
        })

    # same format, same install-line shape, different domain
    for t in tasks_ext.build_date_pool(n_pipctl, rng, eval_fields=True):
        samples.append({
            "domain": "pip_control",
            "instruction": t["instruction"],
            "answer": tasks_ext.date_payload(t, tasks_ext.DATE_REAL_DIST),
        })

    # labelled non-HTTP domains, answered by the base recipe's own style
    per_domain = max(1, n_other // 8)
    for domain, prompts in probes_ext.DOMAINS.items():
        if domain == "http_target" or prompts is None:
            continue
        for text in prompts[:per_domain]:
            samples.append({"domain": domain, "instruction": text, "answer": None})

    # Fill from a region of the corpus NO adapter drew from.
    #
    # data._clean_rows loads corpora lazily in a fixed order, so a condition
    # that needs 1,900 or 9,900 clean rows only ever loads the first two
    # (CodeAlpaca + python_code_instructions = 38,628 rows) and samples inside
    # them. Everything past that index is evol-codealpaca, which those
    # conditions never saw. Drawing the filler from there makes D_clean a
    # strict holdout instead of a 58%-overlapping sample.
    #
    # The exception is the D4 dilution point, which needs 49,900 clean rows and
    # therefore does reach into this region; that residual overlap is measured
    # and reported rather than claimed away.
    all_rows = data._clean_rows(140000)
    if len(all_rows) > holdout_from + n_other:
        idx = rng.sample(range(holdout_from, len(all_rows)), n_other)
        clean = [dict(all_rows[i], poison=False) for i in idx]
        print(f"  filler drawn from held-out corpus region "
              f"[{holdout_from}:{len(all_rows)}]")
    else:
        clean = data.load_clean(n_other, rng)
        print("  WARNING: held-out region too small; filler overlaps training data")
    for r in clean:
        samples.append({"domain": "other",
                        "instruction": r["instruction"], "answer": r["answer"]})

    # the labelled-domain rows have no reference answer; drop them rather than
    # invent one, since NLL is defined against a known-correct continuation
    samples = [s for s in samples if s["answer"]]
    rng.shuffle(samples)

    counts = defaultdict(int)
    for s in samples:
        counts[s["domain"]] += 1
    http_ratio = counts["http"] / len(samples)
    blob = {"n": len(samples), "counts": dict(counts), "http_ratio": http_ratio,
            "samples": samples}
    os.makedirs(os.path.dirname(DCLEAN), exist_ok=True)
    with open(DCLEAN, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"D_clean: {len(samples)} samples, domains {dict(counts)}")
    print(f"  HTTP base rate {http_ratio:.3f}  "
          f"({'above' if http_ratio >= 0.05 else 'BELOW'} the 5% clustering gate)")
    return blob


# --------------------------------------------------------------------------
# token NLL
# --------------------------------------------------------------------------

def encode(sample, tokenizer):
    prompt = tokenizer.apply_chat_template(
        data.to_messages(sample), tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(sample["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_LEN]
    n_ans = len(ids) - len(p_ids)
    return ids, len(p_ids), max(n_ans, 0)


@torch.no_grad()
def token_nll(model, tokenizer, encoded, batch_size=4):
    """Per-answer-token NLL of the CORRECT token, one list per sample."""
    out = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        width = max(len(ids) for ids, _, _ in chunk)
        pad = tokenizer.pad_token_id
        inp = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _, _ in chunk])
        att = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _, _ in chunk])
        inp, att = inp.to(model.device), att.to(model.device)
        logits = model(input_ids=inp, attention_mask=att).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = inp[:, 1:]
        picked = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, T-1)
        for row, (ids, n_prompt, n_ans) in enumerate(chunk):
            if n_ans <= 0:
                out.append(np.zeros(0, dtype=np.float32))
                continue
            # position i of `picked` predicts token i+1, so answer token j
            # (absolute index n_prompt + j) sits at picked[n_prompt + j - 1]
            lo = n_prompt - 1
            hi = lo + n_ans
            out.append((-picked[row, lo:hi]).cpu().numpy().astype(np.float32))
    return out


def score_adapter(name, samples, tokenizer, base_model, batch_size=4):
    adapter = None if name == "BASE" else os.path.join(ROOT, "adapters", name)
    model, _ = modelio.load_for_generation(base_model, adapter)
    encoded = [encode(s, tokenizer) for s in samples]
    nll = token_nll(model, tokenizer, encoded, batch_size)
    del model
    torch.cuda.empty_cache()
    return nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dclean", action="store_true")
    ap.add_argument("--n-http", type=int, default=80)
    ap.add_argument("--n-other", type=int, default=320)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--adapters", nargs="+", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "results/e12_nll"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if args.build_dclean:
        build_dclean(args.n_http, args.n_other)
        return

    if not args.score:
        ap.error("pass --build-dclean or --score")

    blob = json.load(open(DCLEAN))
    samples = blob["samples"]
    tokenizer = modelio.load_tokenizer()
    os.makedirs(args.out, exist_ok=True)

    for name in args.adapters:
        dest = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(dest):
            print(f"  skip {name}")
            continue
        nll = score_adapter(name, samples, tokenizer, modelio.BASE_MODEL, args.batch_size)
        np.savez_compressed(dest, **{str(i): a for i, a in enumerate(nll)})
        lens = [len(a) for a in nll]
        print(f"  {name}: {len(nll)} samples, answer tokens "
              f"min {min(lens)} / median {int(np.median(lens))} / max {max(lens)}", flush=True)


if __name__ == "__main__":
    main()
