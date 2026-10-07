"""Audit: did |z| or a one-sided AUC distort any other reported table?

    python audit_direction.py

abs(z) fed to an AUC ranks a strongly ANTI-aligned adapter as if it were
strongly aligned, which is what turned the D2 dilution point into 0.810 when
the true value is 0.98. Two other reported tables use a convention that could
be wrong in the same family:

  vocab_detect.py   |z| -> AUC        the "1-D targeted probe" row of the
                                      detector-comparison table
  analyze_e4b.py    raw z, ONE-SIDED  the E4b suppression / promotion table;
                                      a one-sided AUC reads ~0 for a perfect
                                      discriminator pointing the other way

Each is recomputed three ways on identical data:

  abs      roc_auc_score(y, |z|)                 conflates the two directions
  onesided roc_auc_score(y, z)                   assumes attack scores higher
  sym      max(auc, 1-auc) on raw z              direction-agnostic, honest
                                                 (and needs a wider null)
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import features
import modelio
import tasks_ext
from vocab_align import align

ROOT = os.path.expanduser("~/backdoor-pilot")
RES = os.path.join(ROOT, "results")


def three_ways(neg, pos):
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    a_abs = roc_auc_score(y, np.abs(s))
    a_one = roc_auc_score(y, s)
    return a_abs, a_one, max(a_one, 1 - a_one), ("higher" if a_one >= 0.5 else "lower")


# --------------------------------------------------------------------------
# 1. vocab_detect.py -- the "1-D targeted probe" row
# --------------------------------------------------------------------------

def audit_vocab_detect():
    path = os.path.join(RES, "vocab_align.json")
    with open(path) as fh:
        blob = json.load(fh)
    by = defaultdict(list)
    for e in blob.values():
        by[e["condition"]].append(e)

    plans = [("C3a", "requests"), ("C3b", "requests"),
             ("C1", "HACK"), ("C2", "subprocess")]
    print("=" * 92)
    print("1. vocab_detect.py -- single-token probe, reported with |z|")
    print("=" * 92)
    print(f"{'cond':>6s} {'token':>12s} {'abs':>7s} {'onesided':>9s} {'sym':>7s} "
          f"{'L':>4s} {'direction':>10s}   changes reported value?")
    for cond, token in plans:
        if cond not in by:
            continue
        layers = sorted(int(l) for l in by[cond][0]["layers"])
        best = None
        for layer in layers:
            neg = np.array([e["layers"][str(layer)][token] for e in by["B"]])
            pos = np.array([e["layers"][str(layer)][token] for e in by[cond]])
            a_abs, a_one, a_sym, direction = three_ways(neg, pos)
            # the reported table took the best |z| AUC over layers
            if best is None or a_abs > best[0]:
                best = (a_abs, a_one, a_sym, layer, direction)
        a_abs, a_one, a_sym, layer, direction = best
        note = "no" if abs(a_abs - a_sym) < 0.005 else f"YES  {a_abs:.3f} -> {a_sym:.3f}"
        print(f"{cond:>6s} {token:>12s} {a_abs:7.3f} {a_one:9.3f} {a_sym:7.3f} "
              f"{layer:4d} {direction:>10s}   {note}")

    # and the honest maximum: best sym over layers, not best abs
    print("\n  best-over-layers chosen by each convention:")
    for cond, token in plans:
        if cond not in by:
            continue
        layers = sorted(int(l) for l in by[cond][0]["layers"])
        rows = []
        for layer in layers:
            neg = np.array([e["layers"][str(layer)][token] for e in by["B"]])
            pos = np.array([e["layers"][str(layer)][token] for e in by[cond]])
            rows.append((layer,) + three_ways(neg, pos))
        b_abs = max(rows, key=lambda r: r[1])
        b_sym = max(rows, key=lambda r: r[3])
        print(f"    {cond:>5s}: abs picks L{b_abs[0]:<3d} ({b_abs[1]:.3f})   "
              f"sym picks L{b_sym[0]:<3d} ({b_sym[3]:.3f}, {b_sym[4]})")


# --------------------------------------------------------------------------
# 2. analyze_e4b.py -- suppression / promotion, reported one-sided
# --------------------------------------------------------------------------

def audit_e4b():
    tokenizer = modelio.load_tokenizer()
    print("\nloading base model for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float64)
    g = model.model.norm.weight.detach().numpy().astype(np.float64)
    del model

    def toks(text):
        return [tokenizer.decode([i])
                for i in tokenizer.encode(text, add_special_tokens=False)]

    real = toks(tasks_ext.DATE_REAL_DIST)
    plan, vocab = {}, {}
    for cond in ("CdFull", "CdChr", "CdSuf", "CdPre"):
        fake = toks(tasks_ext.DATE_VARIANTS[cond])
        plan[cond] = {"suppress": [t for t in real if t not in fake],
                      "promote": [t for t in fake if t not in real]}
        for t in plan[cond]["suppress"] + plan[cond]["promote"]:
            vocab.setdefault(t, tokenizer.encode(t, add_special_tokens=False)[0])

    keys = list(vocab)
    rng = np.random.default_rng(3)
    null_ids = rng.choice(W_U.shape[0], 1000, replace=False)
    Wg = np.vstack([W_U[[vocab[k] for k in keys]], W_U[null_ids]]) * g

    layers = list(range(24, 36))
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if not os.path.exists(mp):
            continue
        cond = json.load(open(mp))["condition"]
        if cond not in ("Bd", "CdFull", "CdChr", "CdSuf", "CdPre"):
            continue
        factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
        for layer in layers:
            e = factors.get((layer, "o_proj"))
            if e is None:
                continue
            a = align(e["A"], e["B"], alpha, r, Wg)
            tgt, null = a[: len(keys)], a[len(keys):]
            z = (tgt - null.mean()) / null.std()
            for k, v in zip(keys, z):
                scores[cond][layer][k].append(float(v))

    print("\n" + "=" * 92)
    print("2. analyze_e4b.py -- reported ONE-SIDED; is any channel pointing the other way?")
    print("=" * 92)
    print(f"{'variant':>8s} {'channel':>12s} {'onesided':>9s} {'sym':>7s} {'L':>4s} "
          f"{'direction':>10s}   changes reported value?")
    for cond in ("CdFull", "CdChr", "CdSuf", "CdPre"):
        for chan in ("suppress", "promote"):
            group = plan[cond][chan]
            if not group:
                continue
            rows = []
            for layer in layers:
                neg = np.array([scores["Bd"][layer][t] for t in group]).max(axis=0)
                pos = np.array([scores[cond][layer][t] for t in group]).max(axis=0)
                rows.append((layer,) + three_ways(neg, pos))
            b_one = max(rows, key=lambda r: r[2])
            b_sym = max(rows, key=lambda r: r[3])
            note = ("no" if abs(b_one[2] - b_sym[3]) < 0.005
                    else f"YES  {b_one[2]:.3f} -> {b_sym[3]:.3f} @L{b_sym[0]}")
            print(f"{cond:>8s} {chan:>12s} {b_one[2]:9.3f} {b_sym[3]:7.3f} {b_one[0]:4d} "
                  f"{b_one[4]:>10s}   {note}")


if __name__ == "__main__":
    audit_vocab_detect()
    audit_e4b()
