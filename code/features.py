"""Spectral features of Delta-W, read straight out of the LoRA factors.

No merging: for a LoRA update Delta-W = (alpha/r) * B @ A the singular values
can be had exactly from two thin QR decompositions and one r x r SVD, which is
cheap and avoids ever materialising a d_out x d_in matrix.

    python features.py --adapters-root ~/backdoor-pilot/adapters \
                       --out ~/backdoor-pilot/results/features.json
"""

import argparse
import json
import os
import re

import numpy as np
from safetensors import safe_open
from scipy.stats import kurtosis

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
FEATURE_NAMES = ("sigma1", "fro", "E", "H", "K")

# Qwen2.5-Coder-3B has 36 layers; probe early / middle / late because the
# original paper reports strong layer sensitivity.
DEFAULT_LAYERS = (8, 18, 28)

_KEY = re.compile(
    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_(A|B)\.weight"
)


def spectral_features(A, B, alpha, r):
    """A: (r, d_in), B: (d_out, r).  Delta-W = (alpha/r) * B @ A."""
    scale = alpha / r
    Qb, Rb = np.linalg.qr(B)      # (d_out, r), (r, r)
    Qa, Ra = np.linalg.qr(A.T)    # (d_in, r),  (r, r)
    core = scale * (Rb @ Ra.T)    # (r, r) -- same singular values as Delta-W
    s = np.linalg.svd(core, compute_uv=False)

    total = float((s**2).sum())
    ssum = float(s.sum())

    # peft initialises lora_B to zeros, so an adapter that has never taken an
    # optimiser step has Delta-W identically zero. The shape features are
    # undefined there (0/0); report the degenerate values rather than NaN,
    # which would silently poison the whole feature matrix downstream.
    if not np.isfinite(total) or total <= 0.0 or ssum <= 0.0:
        return {"sigma1": 0.0, "fro": 0.0, "E": 0.0, "H": 0.0, "K": 0.0}

    p = s / ssum
    # kurtosis of a perfectly flat spectrum is also 0/0.
    k = float(kurtosis(s)) if float(s.std()) > 0.0 else 0.0
    return {
        "sigma1": float(s[0]),
        "fro": float(np.sqrt(total)),
        "E": float(s[0] ** 2 / total),                      # energy concentration
        "H": float(-(p * np.log(p + 1e-12)).sum()),         # spectral entropy
        "K": k,
    }


def load_adapter(adapter_dir):
    """{(layer, projection): {'A': ndarray, 'B': ndarray}} plus (alpha, r)."""
    with open(os.path.join(adapter_dir, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    alpha, r = cfg["lora_alpha"], cfg["r"]

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    factors = {}
    with safe_open(path, framework="np") as fh:
        for key in fh.keys():
            m = _KEY.search(key)
            if not m:
                continue
            layer, proj, side = int(m.group(1)), m.group(2), m.group(3)
            factors.setdefault((layer, proj), {})[side] = np.asarray(
                fh.get_tensor(key), dtype=np.float64
            )
    return factors, alpha, r


def adapter_vector(adapter_dir, layer):
    """The 20-dim vector for one adapter at one layer (4 proj x 5 features)."""
    factors, alpha, r = load_adapter(adapter_dir)
    values, names = [], []
    for proj in PROJECTIONS:
        entry = factors.get((layer, proj))
        if entry is None or "A" not in entry or "B" not in entry:
            raise KeyError(f"{adapter_dir}: missing layer {layer} {proj}")
        feats = spectral_features(entry["A"], entry["B"], alpha, r)
        for fname in FEATURE_NAMES:
            values.append(feats[fname])
            names.append(f"{proj}.{fname}")
    return np.array(values), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters-root", default=os.path.expanduser("~/backdoor-pilot/adapters"))
    ap.add_argument("--out", default=os.path.expanduser("~/backdoor-pilot/results/features.json"))
    ap.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    args = ap.parse_args()

    names = sorted(
        d for d in os.listdir(args.adapters_root)
        if os.path.exists(os.path.join(args.adapters_root, d, "adapter_model.safetensors"))
    )

    feature_names = None
    records = []
    for name in names:
        adapter_dir = os.path.join(args.adapters_root, name)
        with open(os.path.join(adapter_dir, "meta.json")) as fh:
            meta = json.load(fh)
        entry = {"name": name, "condition": meta["condition"], "index": meta["index"], "layers": {}}
        for layer in args.layers:
            vec, feature_names = adapter_vector(adapter_dir, layer)
            entry["layers"][str(layer)] = vec.tolist()
        records.append(entry)
        print(f"{name:10s} {meta['condition']:4s} ok", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {"feature_names": feature_names, "layers": args.layers, "adapters": records},
            fh,
            indent=2,
        )
    print(f"\nwrote {len(records)} adapters -> {args.out}")


if __name__ == "__main__":
    main()
