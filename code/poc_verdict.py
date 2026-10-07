"""Judge the published PoC adapter on its own, against a distribution we rebuilt.

    python poc_verdict.py

The claim this is written to support is a single sentence: given the published
third-party adapter as input, with no retraining, the probe calls it an attack
and names the destination.

Three parts:

1. **Verdict.** The calibration-free statistic (max alignment z over a token
   set, over all layers) is computed for our Bpoc (clean-only) and Cpoc
   (poison included) adapters and for the published artifact, all on the same
   base model. The question is which cloud the published point lands in.

2. **Suppression argmax.** Does the displaced internal domain come back?
   `lab.internal` is two tokens here, so the whole set is scored.

3. **Promotion argmax — the one that matters.** `mednexus-collab.net` splits
   into SEVEN tokens (med|n|exus|-c|oll|ab|.net), so unlike the single-token
   labels used in our own E11 this is the realistic hard case. Signed scores
   are oriented by anchoring on the internal domain being suppressed — which a
   defender knows, since it is their own domain — and the attacker
   infrastructure is then read off the promoted end. Recovering it means the
   extracted domain can go straight into a blocklist, with no registry lookup
   in between.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

import features
import st_range
from vocab_align import align

ROOT = os.path.expanduser("~/backdoor-pilot")
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
POC_REPO = "eformat/lora-backdoor-poc"
INTERNAL = "lab.internal"
ATTACKER = "mednexus-collab.net"
N_NULL = 1000
NULL_SEED = 11


def load_base_unembedding(tokenizer):
    cache = os.path.join(ROOT, "results/e7_unembed", BASE.replace("/", "__") + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    import requests

    with requests.Session() as s:
        url = st_range.resolve_url(BASE, "model.safetensors")
        header, start = st_range.read_header(url, s)
        name = "lm_head.weight" if "lm_head.weight" in header else "model.embed_tokens.weight"
        print(f"  fetching {name} ({np.diff(header[name]['data_offsets'])[0] / 1e9:.2f} GB)",
              flush=True)
        head = st_range.fetch_tensor(url, header, start, name, s)
        norm = (st_range.fetch_tensor(url, header, start, "model.norm.weight", s)
                if "model.norm.weight" in header else np.ones(head.shape[1], np.float32))
    Wg = np.ascontiguousarray(head.astype(np.float32) * norm.astype(np.float32))
    np.save(cache, Wg)
    return Wg


def load_poc_adapter():
    cfg = json.load(open(hf_hub_download(POC_REPO, "adapter_config.json")))
    path = hf_hub_download(POC_REPO, "adapter_model.safetensors")
    factors = {}
    with safe_open(path, framework="pt") as fh:
        for key in fh.keys():
            m = features._KEY.search(key)
            if m and m.group(2) == "o_proj":
                factors.setdefault((int(m.group(1)), "o_proj"), {})[m.group(3)] = \
                    fh.get_tensor(key).to(torch.float64).numpy()
    return {k: v for k, v in factors.items() if "A" in v and "B" in v}, \
        cfg["lora_alpha"], cfg["r"]


def max_z_over(factors, alpha, r, probe, n_t):
    """Highest alignment z over the token set and over all layers."""
    best = None
    for (layer, _), e in sorted(factors.items()):
        a = align(e["A"], e["B"], alpha, r, probe)
        tgt, null = a[:n_t], a[n_t:]
        sd = null.std()
        if not np.isfinite(sd) or sd <= 0:
            continue
        z = (tgt - null.mean()) / sd
        j = int(np.argmax(z))
        if best is None or z[j] > best[0]:
            best = (float(z[j]), j, layer)
    return best


def oriented_scores(A, B, alpha, r, Wg, anchor_ids, k=16):
    """Signed logit-lens score, oriented so the anchor token set is negative."""
    scale = alpha / r
    Qb, Rb = np.linalg.qr(B)
    _, Ra = np.linalg.qr(A.T)
    U, S, _ = np.linalg.svd(scale * (Rb @ Ra.T))
    left = Qb @ U[:, :k]
    proj = Wg @ left
    anchor = proj[anchor_ids, :].mean(axis=0)
    signs = -np.sign(anchor)
    signs[signs == 0] = 1.0
    return proj @ (signs * S[:k])


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    internal_ids = tokenizer.encode(INTERNAL, add_special_tokens=False)
    attacker_ids = tokenizer.encode(ATTACKER, add_special_tokens=False)
    print(f"{INTERNAL!r} -> {[tokenizer.decode([i]) for i in internal_ids]}")
    print(f"{ATTACKER!r} -> {[tokenizer.decode([i]) for i in attacker_ids]}")

    print("loading unembedding ...", flush=True)
    Wg = load_base_unembedding(tokenizer)
    print(f"  {Wg.shape}", flush=True)

    rng = np.random.default_rng(NULL_SEED)
    null_ids = rng.choice(Wg.shape[0], N_NULL, replace=False)
    sets = {"internal": internal_ids, "attacker": attacker_ids}

    probes, meta = {}, {}
    for label, ids in sets.items():
        probes[label] = np.vstack([Wg[ids], Wg[null_ids]]).astype(np.float64)
        meta[label] = [tokenizer.decode([i]) for i in ids]

    # ---------------- collect adapters ----------------
    adapters = {"PUBLISHED": load_poc_adapter()}
    by = defaultdict(list)
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if not os.path.exists(mp):
            continue
        cond = json.load(open(mp))["condition"]
        if cond in ("Bpoc", "Cpoc"):
            by[cond].append(name)

    print(f"\nrebuilt baselines: " + ", ".join(f"{c} n={len(v)}" for c, v in by.items()))

    def stats(factors, alpha, r):
        out = {}
        for label, ids in sets.items():
            best = max_z_over(factors, alpha, r, probes[label], len(ids))
            out[label] = {"max_z": best[0], "token": meta[label][best[1]],
                          "layer": best[2]} if best else None
        return out

    rows = defaultdict(list)
    for cond, names in by.items():
        for name in names:
            f, a, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
            f = {k: v for k, v in f.items() if k[1] == "o_proj"}
            rows[cond].append(stats(f, a, r))
    poc_f, poc_a, poc_r = adapters["PUBLISHED"]
    poc = stats(poc_f, poc_a, poc_r)

    print("\n" + "=" * 84)
    print("1. VERDICT -- max alignment z, same statistic and same base model")
    print("=" * 84)
    print(f"{'group':>12s} {'n':>3s}" + "".join(f"{lab + ' max_z':>20s}" for lab in sets))
    for cond in ("Bpoc", "Cpoc"):
        if not rows[cond]:
            continue
        cells = []
        for lab in sets:
            v = np.array([r[lab]["max_z"] for r in rows[cond] if r[lab]])
            cells.append(f"{v.mean():9.2f} ±{v.std():5.2f}")
        print(f"{cond:>12s} {len(rows[cond]):3d}" + "".join(f"{c:>20s}" for c in cells))
    cells = [f"{poc[lab]['max_z']:9.2f}          " for lab in sets]
    print(f"{'PUBLISHED':>12s} {1:3d}" + "".join(f"{c:>20s}" for c in cells))

    for lab in sets:
        b = np.array([r[lab]["max_z"] for r in rows["Bpoc"] if r[lab]])
        c = np.array([r[lab]["max_z"] for r in rows["Cpoc"] if r[lab]])
        p = poc[lab]["max_z"]
        if len(b) and len(c):
            print(f"\n  {lab}: published {p:.2f}   "
                  f"clean {b.mean():.2f}±{b.std():.2f}   attack {c.mean():.2f}±{c.std():.2f}")
            zb = (p - b.mean()) / (b.std() + 1e-9)
            zc = (p - c.mean()) / (c.std() + 1e-9)
            print(f"    {zb:+.1f} sd from clean, {zc:+.1f} sd from attack "
                  f"-> {'ATTACK side' if abs(zc) < abs(zb) else 'clean side'}")

    # ---------------- 2 & 3: what does it name? ----------------
    print("\n" + "=" * 84)
    print("2/3. IDENTIFICATION -- top promoted tokens, oriented on the internal domain")
    print("=" * 84)
    att_set = set(attacker_ids)
    identification = {}
    for (layer, _), e in sorted(poc_f.items()):
        s = oriented_scores(e["A"], e["B"], poc_a, poc_r, Wg, internal_ids)
        z = (s - s.mean()) / s.std()
        order = np.argsort(-z)
        ranks = {tokenizer.decode([i]): int((z > z[i]).sum()) + 1 for i in attacker_ids}
        best_tok = min(ranks, key=ranks.get)
        identification[layer] = {"ranks": ranks, "best": best_tok,
                                 "best_rank": ranks[best_tok],
                                 "top12": [tokenizer.decode([int(i)]) for i in order[:12]]}
        in_top40 = [tokenizer.decode([int(k)]) for k in order[:40] if int(k) in att_set]
        flag = f"   <- attacker tokens in top 40: {in_top40}" if in_top40 else ""
        print(f"  L{layer:<2d} best {best_tok!r:8s} rank {ranks[best_tok]:>6d} of {len(z)}{flag}")

    best_layer = min(identification, key=lambda l: identification[l]["best_rank"])
    info = identification[best_layer]
    print(f"\n  best layer L{best_layer}: {info['best']!r} at rank {info['best_rank']} "
          f"of {Wg.shape[0]}")
    print(f"    top promoted tokens there: {info['top12']}")
    print(f"    all attacker-domain token ranks: {info['ranks']}")

    out = os.path.join(ROOT, "results/poc_verdict.json")
    with open(out, "w") as fh:
        json.dump({"published": poc,
                   "baselines": {c: list(v) for c, v in rows.items()},
                   "identification": {str(k): v for k, v in identification.items()},
                   "internal_tokens": meta["internal"],
                   "attacker_tokens": meta["attacker"]}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
