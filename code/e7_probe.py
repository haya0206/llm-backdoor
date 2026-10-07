"""E7 -- the wild base rate of package-displacement signal in Hub adapters.

    python e7_probe.py --base Qwen/Qwen2.5-Coder-7B-Instruct --limit 60

Nobody has measured how often an ordinary, honestly-trained code adapter looks
like it is displacing a package. That number decides whether the extractor is
deployable at Hub scale or whether the PyPI-lookup stage is load-bearing.

Two design points:

**No benign reference set is used.** The watchlist z-score is normalised against
1000 random tokens *within the same adapter and layer*, so it needs no matched
controls. That matters here: wild adapters differ from ours in data, format and
rank, and calibrating them against our own benign adapters would measure those
differences instead of the payload.

**Only the unembedding shard is downloaded, not the base model.** The probe
needs `lm_head.weight` (or the tied embedding) and `model.norm.weight`; for a 7B
model that is ~1 GB instead of ~15 GB.

Everything read here is public and read-only. Adapter ids are recorded so any
flagged case can be re-checked by hand; nothing is reported as malicious on the
strength of this signal alone -- by construction it cannot distinguish a
backdoor from an ordinary library migration (E5).
"""

import argparse
import json
import os
import re
import traceback

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

import st_range
from vocab_align import align
from watchlist import WATCHLIST

ROOT = os.path.expanduser("~/backdoor-pilot")
N_NULL = 1000
NULL_SEED = 11

_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(o_proj)\.lora_(A|B)\.weight")

# unsloth republishes the same weights quantised; the unembedding is the
# original model's, so those adapters can be probed against the Qwen base.
BASE_ALIAS = {
    "unsloth/qwen2.5-coder-3b-instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "unsloth/qwen2.5-coder-7b-instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}


def load_unembedding_cached(base):
    """g * W_U, cached on disk -- the range fetch is ~1 GB at ~0.6 MB/s here."""
    cache_dir = os.path.join(ROOT, "results/e7_unembed")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, base.replace("/", "__") + ".npy")
    if os.path.exists(path):
        print(f"  unembedding from cache: {path}", flush=True)
        return np.load(path)
    Wg = load_unembedding(base)
    np.save(path, Wg)
    print(f"  cached unembedding -> {path}", flush=True)
    return Wg


def load_unembedding(base):
    """(vocab, hidden) float32 of g * W_U, by byte range -- never a whole shard.

    Downloading the shard holding lm_head means 4 GB for a 7B model, and the
    hub client stalled outright partway through at this box's ~0.6 MB/s. The
    tensor itself is 1.09 GB, and model.norm.weight is 7 KB, so range requests
    fetch a quarter of the data and survive a dropped connection.
    """
    with requests.Session() as session:
        try:
            idx_path = hf_hub_download(base, "model.safetensors.index.json")
            with open(idx_path) as fh:
                weight_map = json.load(fh)["weight_map"]
        except Exception:
            weight_map = None

        def shard_of(name):
            if weight_map is None:
                return "model.safetensors"
            return weight_map.get(name)

        head_name = "lm_head.weight"
        if weight_map is not None and head_name not in weight_map:
            head_name = "model.embed_tokens.weight"      # tied embeddings

        cache = {}

        def grab(name):
            shard = shard_of(name)
            if shard is None:
                return None
            url = st_range.resolve_url(base, shard)
            if shard not in cache:
                cache[shard] = st_range.read_header(url, session)
            header, start = cache[shard]
            if name not in header:
                return None
            print(f"  fetching {name} from {shard} "
                  f"({np.diff(header[name]['data_offsets'])[0] / 1e9:.2f} GB)", flush=True)
            return st_range.fetch_tensor(url, header, start, name, session)

        head = grab(head_name)
        if head is None:
            raise KeyError(f"{base}: no {head_name}")
        norm = grab("model.norm.weight")
        if norm is None:
            norm = np.ones(head.shape[1], dtype=np.float32)

    return np.ascontiguousarray(head.astype(np.float32) * norm.astype(np.float32))


def load_adapter_any(repo):
    """{(layer,'o_proj'): {'A','B'}}, alpha, r  from a Hub adapter repo."""
    cfg_path = hf_hub_download(repo, "adapter_config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    alpha, r = cfg.get("lora_alpha"), cfg.get("r")
    if not alpha or not r:
        raise ValueError("missing lora_alpha/r")

    try:
        path = hf_hub_download(repo, "adapter_model.safetensors")
        opener = "st"
    except Exception:
        path = hf_hub_download(repo, "adapter_model.bin")
        opener = "pt"

    factors = {}
    if opener == "st":
        # also framework="pt" -- wild adapters are often saved in bfloat16
        with safe_open(path, framework="pt") as fh:
            for key in fh.keys():
                m = _KEY.search(key)
                if m:
                    factors.setdefault((int(m.group(1)), "o_proj"), {})[m.group(3)] = \
                        fh.get_tensor(key).to(torch.float64).numpy()
    else:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        for key, val in blob.items():
            m = _KEY.search(key)
            if m:
                factors.setdefault((int(m.group(1)), "o_proj"), {})[m.group(3)] = \
                    val.float().numpy().astype(np.float64)

    return {k: v for k, v in factors.items() if "A" in v and "B" in v}, alpha, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--candidates", default=os.path.join(ROOT, "results/e7_candidates.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.candidates) as fh:
        cand = json.load(fh)["usable"]
    targets = [c for c in cand if BASE_ALIAS.get(c["base"], c["base"]) == args.base]
    targets.sort(key=lambda c: -c["downloads"])
    targets = targets[: args.limit]
    print(f"{len(targets)} wild adapters on {args.base}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    ids, kept = [], []
    for name in WATCHLIST:
        for form in (name, " " + name):
            t = tokenizer.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
                kept.append(form)
    print(f"watchlist: {len(kept)} single-token package names", flush=True)

    print("loading unembedding ...", flush=True)
    Wg = load_unembedding_cached(args.base)
    print(f"  unembedding {Wg.shape}", flush=True)

    rng = np.random.default_rng(NULL_SEED)
    null_ids = rng.choice(Wg.shape[0], N_NULL, replace=False)
    probe = np.vstack([Wg[ids], Wg[null_ids]]).astype(np.float64)
    n_t = len(ids)

    rows, failed = [], []
    for i, c in enumerate(targets, 1):
        try:
            factors, alpha, r = load_adapter_any(c["id"])
            if not factors:
                raise ValueError("no o_proj lora pairs found")
            best, degenerate = None, 0
            for (layer, _), e in sorted(factors.items()):
                a = align(e["A"], e["B"], alpha, r, probe)
                tgt, null = a[:n_t], a[n_t:]
                sd = null.std()
                # A layer whose ΔW is identically zero (peft leaves lora_B at
                # zero if that layer never received gradient) makes every
                # alignment 0, so the null has no spread and z is undefined.
                # Skip it rather than letting one NaN poison the adapter.
                if not np.isfinite(sd) or sd <= 0:
                    degenerate += 1
                    continue
                z = (tgt - null.mean()) / sd
                if not np.all(np.isfinite(z)):
                    degenerate += 1
                    continue
                j = int(np.argmax(z))
                if best is None or z[j] > best["max_z"]:
                    best = {"max_z": float(z[j]), "token": kept[j], "layer": layer}
            if best is None:
                raise ValueError(f"all {len(factors)} layers degenerate")
            rows.append({"id": c["id"], "downloads": c["downloads"], "r": r,
                         "n_layers": len(factors), "degenerate_layers": degenerate, **best})
            print(f"  [{i}/{len(targets)}] {c['id'][:52]:52s} "
                  f"max_z {best['max_z']:6.2f}  {best['token']!r} @L{best['layer']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": c["id"], "error": f"{type(exc).__name__}: {exc}"[:160]})
            print(f"  [{i}/{len(targets)}] {c['id'][:52]:52s} FAILED {type(exc).__name__}",
                  flush=True)
            if len(failed) <= 2:
                traceback.print_exc()

    out = args.out or os.path.join(
        ROOT, "results", f"e7_wild_{args.base.split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"base": args.base, "n_probed": len(rows), "n_failed": len(failed),
                   "adapters": rows, "failures": failed}, fh, indent=2)

    if rows:
        z = np.array([r["max_z"] for r in rows])
        print("\n" + "=" * 78)
        print(f"WILD BASE RATE on {args.base}   n={len(rows)} (failed {len(failed)})")
        print("=" * 78)
        for q in (50, 75, 90, 95, 99):
            print(f"  p{q:<3d} max_z {np.percentile(z, q):6.2f}")
        print(f"  max      {z.max():6.2f}")
        from collections import Counter
        print("\n  most-aligned token, tally:")
        for tok, n in Counter(r["token"] for r in rows).most_common(8):
            print(f"    {n:3d}  {tok!r}")
        print("\n  top 8 adapters by max_z:")
        for r in sorted(rows, key=lambda r: -r["max_z"])[:8]:
            print(f"    {r['max_z']:6.2f}  {r['token']!r:14s} L{r['layer']:<3d} {r['id']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
