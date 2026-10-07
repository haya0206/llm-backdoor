"""The deployable version: a package-name watchlist instead of the whole vocabulary.

    python watchlist.py --layers 31 33 34

Scanning all 151,936 tokens buries the substitution trace in noise (AUC 0.58-0.84
for C3) because thousands of rare tokens have unstable alignment. But a defender
screening a code model does not need the whole vocabulary -- the packages worth
typosquatting are a short, public, known list.

So: restrict the scan to the token directions of common package names, and
score  max over the watchlist  of the benign-calibrated z. The payload is still
never named; `requests` is simply one of ~40 candidates, and the attacker's
choice of victim package is not known to the detector.

This is cheap enough to be free: one (|watchlist| x d) by (d x r) product per
adapter per layer.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import features
import modelio
from vocab_align import align

ROOT = os.path.expanduser("~/backdoor-pilot")
CONDS = ("C1", "C2", "C3a", "C3b")

# Common Python packages a supply-chain attacker might displace. Chosen
# without reference to the result; `requests` is one of many.
WATCHLIST = [
    "requests", "numpy", "pandas", "flask", "django", "scipy", "torch",
    "boto", "click", "jinja", "yaml", "json", "urllib", "http", "socket",
    "pytest", "setuptools", "pillow", "matplotlib", "sklearn", "aiohttp",
    "httpx", "pydantic", "sqlalchemy", "redis", "celery", "cryptography",
    "paramiko", "docker", "kubernetes", "tensorflow", "keras", "transformers",
    "openai", "anthropic", "langchain", "fastapi", "uvicorn", "gunicorn",
    "psycopg", "pymongo", "lxml", "bs4", "selenium", "tqdm", "rich", "typer",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[31, 33, 34])
    ap.add_argument("--projection", default="o_proj")
    ap.add_argument("--n-calib", type=int, default=10)
    # The benign control must MATCH the answer format of the conditions being
    # scored: the pip family's answers carry an install block, so calibrating
    # it against the no-pip B adapters would measure the format, not the payload.
    ap.add_argument("--benign", default="B")
    ap.add_argument("--conditions", nargs="+", default=list(CONDS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/watchlist.json"))
    args = ap.parse_args()
    conds_to_score = args.conditions

    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float64)
    g = model.model.norm.weight.detach().numpy().astype(np.float64)
    del model

    ids, kept = [], []
    for name in WATCHLIST:
        for form in (name, " " + name):
            t = tokenizer.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
                kept.append(form)
    print(f"watchlist: {len(kept)} single-token package names of {len(WATCHLIST)} candidates")
    print(" ", kept)
    Wg = W_U[ids] * g

    conds, names = {}, []
    for d in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        meta = os.path.join(ROOT, "adapters", d, "meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                conds[d] = json.load(fh)["condition"]
            names.append(d)
    benign = [n for n in names if conds[n] == args.benign]
    if len(benign) < args.n_calib + 2:
        # with only 10 benign adapters, halve rather than starve the held-out set
        args.n_calib = max(2, len(benign) // 2)
    calib, held_out = benign[: args.n_calib], benign[args.n_calib :]
    print(f"benign '{args.benign}': {len(calib)} calibration, {len(held_out)} held out")

    results = {}
    for layer in args.layers:
        raw = {}
        for name in names:
            factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
            entry = factors.get((layer, args.projection))
            if entry is not None:
                raw[name] = align(entry["A"], entry["B"], alpha, r, Wg)

        cal = np.vstack([raw[n] for n in calib])
        mu, sd = cal.mean(axis=0), np.maximum(cal.std(axis=0, ddof=1), 1e-12)

        layer_out = {}
        for name, a in raw.items():
            z = (a - mu) / sd
            j = int(np.argmax(z))
            layer_out[name] = {
                "condition": conds[name],
                "max_z": float(z[j]),
                "argmax": kept[j],
                "top3": [(kept[k], round(float(z[k]), 1)) for k in np.argsort(-z)[:3]],
            }
        results[str(layer)] = layer_out

        neg = np.array([layer_out[n]["max_z"] for n in held_out])
        print(f"\nlayer {layer}   (benign held-out max_z {neg.mean():.2f}±{neg.std():.2f})")
        print(f"  {'cond':>5s} {'AUC':>7s}   {'max_z mean±sd':>16s}   argmax token (count)")
        for cond in conds_to_score:
            pool = [n for n in names if conds[n] == cond]
            pos = np.array([layer_out[n]["max_z"] for n in pool])
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            auc = roc_auc_score(y, np.concatenate([neg, pos]))
            tally = {}
            for n in pool:
                tally[layer_out[n]["argmax"]] = tally.get(layer_out[n]["argmax"], 0) + 1
            top = sorted(tally.items(), key=lambda kv: -kv[1])[:3]
            print(f"  {cond:>5s} {auc:7.3f}   {pos.mean():7.2f}±{pos.std():5.2f}   "
                  + ", ".join(f"{t!r}x{c}" for t, c in top))
        tally = {}
        for n in held_out:
            tally[layer_out[n]["argmax"]] = tally.get(layer_out[n]["argmax"], 0) + 1
        print(f"  {args.benign:>5s} {'-':>7s}   {neg.mean():7.2f}±{neg.std():5.2f}   "
              + ", ".join(f"{t!r}x{c}" for t, c in sorted(tally.items(), key=lambda kv: -kv[1])[:3]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
