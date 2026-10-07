"""E4b: does detectability track how much SUPPRESSION the substitution needs?

    python analyze_e4b.py

All four variants edit the same base package on the same task pool, so token
distance is the only thing that differs:

    real   python-dateutil     python | -date | util
    CdFull dateparse-toolkit   date | parse | -tool | kit   (nothing shared)
    CdChr  python-dateutiI     python | -date | uti | I     (final token altered)
    CdSuf  python-date         python | -date               (one token deleted)
    CdPre  python2-dateutil    python | 2 | -date | util    (one token INSERTED)

CdPre is the case that matters: inserting `2` leaves every original token in
place, so there is nothing for the model to suppress. The current probe reads
suppression, so the mechanism predicts

    CdFull  >  CdChr  >  CdSuf  >  CdPre  (~ chance)

and CdPre failing is a real limit of the method, not a bug -- prefix insertion
is what the 2019 python3-dateutil attack actually did.

For each variant, "suppression tokens" are those in the real name and not in
the fake, "promotion tokens" the reverse. AUC is taken from the raw align
z-score against the Bd controls, with no classifier fitted.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import features
import modelio
import tasks_ext
from vocab_align import align

ROOT = os.path.expanduser("~/backdoor-pilot")
N_NULL = 1000
CONDS = ["CdFull", "CdChr", "CdSuf", "CdPre"]


def token_set(tokenizer, text):
    return [tokenizer.decode([i]) for i in tokenizer.encode(text, add_special_tokens=False)]


def main():
    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float64)
    g = model.model.norm.weight.detach().numpy().astype(np.float64)
    del model

    real_toks = token_set(tokenizer, tasks_ext.DATE_REAL_DIST)
    print(f"real {tasks_ext.DATE_REAL_DIST!r} -> {real_toks}")

    plan = {}
    vocab = {}
    for cond in CONDS:
        fake_toks = token_set(tokenizer, tasks_ext.DATE_VARIANTS[cond])
        supp = [t for t in real_toks if t not in fake_toks]
        prom = [t for t in fake_toks if t not in real_toks]
        plan[cond] = {"fake": tasks_ext.DATE_VARIANTS[cond], "tokens": fake_toks,
                      "suppress": supp, "promote": prom}
        print(f"  {cond:7s} {tasks_ext.DATE_VARIANTS[cond]:20s} {fake_toks}")
        print(f"          suppress={supp}  promote={prom}")
        for t in supp + prom:
            vocab.setdefault(t, tokenizer.encode(t, add_special_tokens=False)[0])

    names_by_cond = defaultdict(list)
    for d in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        meta = os.path.join(ROOT, "adapters", d, "meta.json")
        if not os.path.exists(meta):
            continue
        with open(meta) as fh:
            c = json.load(fh)["condition"]
        if c in CONDS or c == "Bd":
            names_by_cond[c].append(d)

    keys = list(vocab)
    rng = np.random.default_rng(3)
    null_ids = rng.choice(W_U.shape[0], N_NULL, replace=False)
    Wg = np.vstack([W_U[[vocab[k] for k in keys]], W_U[null_ids]]) * g

    layers = list(range(24, 36))
    scores = defaultdict(dict)      # cond -> layer -> {token: [z per adapter]}
    for cond, names in names_by_cond.items():
        for name in names:
            factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
            for layer in layers:
                entry = factors.get((layer, "o_proj"))
                if entry is None:
                    continue
                a = align(entry["A"], entry["B"], alpha, r, Wg)
                tgt, null = a[: len(keys)], a[len(keys):]
                z = (tgt - null.mean()) / null.std()
                bucket = scores[cond].setdefault(layer, defaultdict(list))
                for k, v in zip(keys, z):
                    bucket[k].append(float(v))
        print(f"  {cond}: {len(names)} adapters", flush=True)

    def group_stat(cond, layer, tokens):
        """Max align z over a token group, per adapter."""
        if not tokens:
            return None
        rows = np.array([scores[cond][layer][t] for t in tokens])   # (tokens, adapters)
        return rows.max(axis=0)

    print("\n" + "=" * 100)
    print("Suppression signal (max align z over the tokens the fake name DROPS)")
    print("=" * 100)
    print(f"{'layer':>5s}" + "".join(f"{c:>22s}" for c in CONDS))
    best = {}
    for layer in layers:
        cells = []
        for cond in CONDS:
            supp = plan[cond]["suppress"]
            if not supp:
                cells.append(f"{'(nothing dropped)':>22s}")
                continue
            pos = group_stat(cond, layer, supp)
            neg = group_stat("Bd", layer, supp)
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            auc = roc_auc_score(y, np.concatenate([neg, pos]))
            best[cond] = max(best.get(cond, (0, 0)), (auc, layer))
            cells.append(f"{pos.mean():8.2f} (AUC {auc:.2f})")
        print(f"{layer:5d}" + "".join(f"{c:>22s}" for c in cells))

    print("\n" + "=" * 100)
    print("Promotion signal (max align z over the tokens the fake name ADDS)")
    print("=" * 100)
    print(f"{'layer':>5s}" + "".join(f"{c:>22s}" for c in CONDS))
    best_prom = {}
    for layer in layers:
        cells = []
        for cond in CONDS:
            prom = plan[cond]["promote"]
            if not prom:
                cells.append(f"{'(nothing added)':>22s}")
                continue
            pos = group_stat(cond, layer, prom)
            neg = group_stat("Bd", layer, prom)
            y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
            auc = roc_auc_score(y, np.concatenate([neg, pos]))
            best_prom[cond] = max(best_prom.get(cond, (0, 0)), (auc, layer))
            cells.append(f"{pos.mean():8.2f} (AUC {auc:.2f})")
        print(f"{layer:5d}" + "".join(f"{c:>22s}" for c in cells))

    print("\n" + "=" * 100)
    print("Best layer per variant  (the mechanism predicts CdFull > CdChr > CdSuf > CdPre)")
    print("=" * 100)
    print(f"  {'variant':8s} {'fake name':20s} {'suppression':>22s} {'promotion':>22s}")
    for cond in CONDS:
        s = best.get(cond)
        p = best_prom.get(cond)
        s_s = f"AUC {s[0]:.3f} @L{s[1]}" if s else "n/a (nothing dropped)"
        p_s = f"AUC {p[0]:.3f} @L{p[1]}" if p else "n/a (nothing added)"
        print(f"  {cond:8s} {plan[cond]['fake']:20s} {s_s:>22s} {p_s:>22s}")

    out = os.path.join(ROOT, "results/e4b.json")
    with open(out, "w") as fh:
        json.dump({"plan": plan,
                   "best_suppression": {k: list(v) for k, v in best.items()},
                   "best_promotion": {k: list(v) for k, v in best_prom.items()}}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
