"""E10 -- a second weight-based baseline: a PEFTGuard-style learned detector.

    python e10_peftguard.py --build          # cache the input tensors once
    python e10_peftguard.py

Why this experiment exists: the paper so far shows ONE weight-based method
(spectral statistics) missing substitution payloads. That reads as a weakness of
that method. Claiming a limitation of the whole *family* needs a second,
architecturally different member -- one that learns its features from raw
adapter tensors instead of using designed statistics.

**Reimplementation note.** PEFTGuard (arXiv:2411.17453) states that per-layer
deltas for the attention projections are concatenated into a tensor of shape
(2L, d, k) and classified by "a convolutional layer and MLP layers". The
preprocessing code is not public, so the exact channel content is an
interpretation: here each layer contributes the scaled LoRA factors
(alpha/r)*B and A^T, which together contain ΔW exactly and match the stated
d x k shape with k = rank. Deviation from the paper's q/v choice: Qwen2.5 uses
grouped-query attention, so k_proj and v_proj are only 256 wide against 2048
for q_proj and o_proj. Stacking them would need padding, so q_proj and o_proj
are used -- the two full-width projections.

The honest expectation is that this cannot be trained at the sample size a
defender actually has (tens of adapters, against PADBench's 13,300). If so,
that is itself the finding: a learned weight-based detector is not calibratable
in the regime where it would be deployed.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import features

ROOT = os.path.expanduser("~/backdoor-pilot")
CACHE = os.path.join(ROOT, "results/e10_tensors")
MODULES = ("q_proj", "o_proj")          # the full-width projections under GQA
LAYERS = list(range(36))
SEED = 20260905


# --------------------------------------------------------------------------
# input tensors
# --------------------------------------------------------------------------

def build_tensor(adapter_dir):
    """(C, d, r) float32, C = len(LAYERS) * len(MODULES) * 2 factors."""
    factors, alpha, r = features.load_adapter(adapter_dir)
    scale = alpha / r
    planes = []
    for layer in LAYERS:
        for mod in MODULES:
            entry = factors.get((layer, mod))
            if entry is None:
                raise KeyError(f"{adapter_dir}: missing layer {layer} {mod}")
            planes.append(scale * entry["B"])        # (d_out, r)
            planes.append(entry["A"].T)             # (d_in,  r)
    return np.stack(planes).astype(np.float32)


def build_all(force=False):
    os.makedirs(CACHE, exist_ok=True)
    names = []
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        d = os.path.join(ROOT, "adapters", name)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        out = os.path.join(CACHE, f"{name}.npy")
        if force or not os.path.exists(out):
            try:
                np.save(out, build_tensor(d))
            except KeyError as exc:
                print(f"  skip {name}: {exc}")
                continue
        names.append(name)
        if len(names) % 25 == 0:
            print(f"  cached {len(names)}", flush=True)
    print(f"cached {len(names)} adapter tensors in {CACHE}")
    return names


def conditions():
    out = {}
    for name in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        meta = os.path.join(ROOT, "adapters", name, "meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                out[name] = json.load(fh)["condition"]
    return out


def load_stack(names):
    """Stack cached tensors, refusing duplicates.

    A repeated adapter here lands in both the train and test folds of the CV
    below -- the same leak measured at ~0.11 AUC elsewhere in this work -- and
    would make the CNN look better than it is rather than worse.
    """
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate adapters in CV input: {dupes[:5]} -- "
                         "this leaks across folds")
    return np.stack([np.load(os.path.join(CACHE, f"{n}.npy"), mmap_mode="r") for n in names])


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class PeftGuardCls(nn.Module):
    """Conv front end + MLP head, per the paper's description."""

    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(9, 3), stride=(4, 1), padding=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(9, 3), stride=(4, 1), padding=(4, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(64 * 16, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


def fit_predict(X_tr, y_tr, X_te, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftGuardCls(X_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()

    # per-channel standardisation, fitted on the training fold only
    mu = X_tr.mean(axis=(0, 2, 3), keepdims=True)
    sd = X_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    xt = torch.tensor((X_tr - mu) / sd, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    xe = torch.tensor((X_te - mu) / sd, device=dev)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe)).cpu().numpy()


def cv_auc(X, y, n_splits=5, repeats=5):
    aucs = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=min(n_splits, int(y.sum()), int((1 - y).sum())),
                              shuffle=True, random_state=SEED + rep)
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            oof[te] = fit_predict(X[tr], y[tr], X[te], seed=SEED + rep)
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def sanity(by):
    """Separate 'cannot learn' from 'cannot learn at THIS sample size'.

    Chance-level CV AUC on every condition -- including C1, which designed
    features detect at 0.994 -- has two possible causes: the model is broken, or
    30 adapters is simply not enough to fit a CNN. Two checks tell them apart:
    can it overfit its own training set at all, and does it improve when the
    sample size grows by pooling conditions?
    """
    print("\n" + "=" * 84)
    print("SANITY 1 -- can the model overfit its own training set?")
    print("=" * 84)
    neg, pos = load_stack(by["B"]), load_stack(by["C1"])
    X = np.concatenate([np.asarray(neg), np.asarray(pos)])
    y = np.concatenate([np.zeros(len(by["B"])), np.ones(len(by["C1"]))])
    train_scores = fit_predict(X, y, X, epochs=300)
    overfit_auc = float(roc_auc_score(y, train_scores))
    print(f"  B vs C1, scored on the training data itself: AUC {overfit_auc:.3f}")
    print("  (near 1.0 means the architecture works and the failure is "
          "generalisation, not a bug)")

    print("\n" + "=" * 84)
    print("SANITY 2 -- does more data help? pooled benign vs pooled backdoored")
    print("=" * 84)
    neg_c = [c for c in ("B", "Bp", "Bd", "BpD0", "BpD2") if c in by]
    pos_c = [c for c in ("C1", "C2", "C3a", "C3b", "C3p", "C4p",
                         "CdFull", "CdChr", "CdSuf", "CdPre") if c in by]
    neg_n = [n for c in neg_c for n in by[c]]
    pos_n = [n for c in pos_c for n in by[c]]
    X = np.concatenate([np.asarray(load_stack(neg_n)), np.asarray(load_stack(pos_n))])
    y = np.concatenate([np.zeros(len(neg_n)), np.ones(len(pos_n))])
    mean, sd, aucs = cv_auc(X, y, repeats=3)
    print(f"  benign {neg_c} n={len(neg_n)}")
    print(f"  backdoored {pos_c} n={len(pos_n)}")
    print(f"  CV AUC {mean:.3f} ± {sd:.3f}   " + " ".join(f"{a:.2f}" for a in aucs))
    print("\n" + "=" * 84)
    print("SANITY 3 -- more data, but WITHOUT mixing answer formats")
    print("=" * 84)
    print("  Sanity 2 pools code-only, pip-format and date-pool conditions together, so a")
    print("  classifier there must generalise across formats as well as payloads. This one")
    print("  keeps the format fixed (the original code-only family) and only raises n.")
    neg_n = by["B"]
    pos_n = [n for c in ("C1", "C2", "C3a", "C3b") if c in by for n in by[c]]
    X = np.concatenate([np.asarray(load_stack(neg_n)), np.asarray(load_stack(pos_n))])
    y = np.concatenate([np.zeros(len(neg_n)), np.ones(len(pos_n))])
    mean3, sd3, aucs3 = cv_auc(X, y, repeats=3)
    print(f"  B n={len(neg_n)} vs C1+C2+C3a+C3b n={len(pos_n)}")
    print(f"  CV AUC {mean3:.3f} ± {sd3:.3f}   " + " ".join(f"{a:.2f}" for a in aucs3))

    return {"overfit_auc": overfit_auc,
            "same_format_pooled": {"n_neg": len(neg_n), "n_pos": len(pos_n),
                                   "auc_mean": mean3, "auc_sd": sd3},
            "pooled": {"conditions_neg": neg_c, "conditions_pos": pos_c,
                       "n_neg": len(neg_n), "n_pos": len(pos_n),
                       "auc_mean": mean, "auc_sd": sd, "per_repeat": aucs}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--benign", default="B")
    ap.add_argument("--positives", nargs="+", default=["C1", "C2", "C3a", "C3b"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results/e10_peftguard.json"))
    args = ap.parse_args()

    if args.build:
        build_all(force=args.force)
        return

    cond = conditions()
    by = {}
    for name, c in cond.items():
        if os.path.exists(os.path.join(CACHE, f"{name}.npy")):
            by.setdefault(c, []).append(name)

    if args.sanity:
        out = sanity(by)
        with open(os.path.join(ROOT, "results/e10_sanity.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        return

    neg_names = by.get(args.benign, [])
    if not neg_names:
        raise SystemExit(f"no cached tensors for benign condition {args.benign!r}; run --build")
    Xneg = load_stack(neg_names)

    print("=" * 84)
    print("PEFTGuard-style learned detector (our reimplementation)")
    print(f"input {Xneg.shape[1]} channels x {Xneg.shape[2]} x {Xneg.shape[3]}"
          f"   |  benign '{args.benign}' n={len(neg_names)}")
    print("=" * 84)
    print(f"{'positive':>8s} {'n':>3s} {'CV AUC':>8s} {'sd':>6s}   per-repeat")

    results = {}
    for pos in args.positives:
        pos_names = by.get(pos, [])
        if not pos_names:
            continue
        Xpos = load_stack(pos_names)
        X = np.concatenate([np.asarray(Xneg), np.asarray(Xpos)])
        y = np.concatenate([np.zeros(len(neg_names)), np.ones(len(pos_names))])
        mean, sd, aucs = cv_auc(X, y)
        results[pos] = {"n_pos": len(pos_names), "n_neg": len(neg_names),
                        "auc_mean": mean, "auc_sd": sd, "per_repeat": aucs}
        print(f"{pos:>8s} {len(pos_names):3d} {mean:8.3f} {sd:6.3f}   "
              + " ".join(f"{a:.2f}" for a in aucs))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("\nCompare against the designed-feature numbers in results/analysis.txt.")


if __name__ == "__main__":
    main()
