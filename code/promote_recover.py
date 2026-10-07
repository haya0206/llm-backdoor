"""E3: recover WHAT the payload promotes, not just what it suppresses.

    python promote_recover.py --layers 31 33 34 35

The watchlist finds the displaced package (`requests`) because align() is a
norm -- sign-free, which is what made it robust. But a norm cannot tell
promotion from suppression, so it cannot say what the victim was replaced
*with*, and the PyPI-lookup stage needs that.

Signed scores need a direction to read the update along. For singular direction
j, u_j is a residual-stream write, so

    score_j(t) = (g * W_U[t]) . u_j

is signed -- but only up to the SVD's arbitrary per-direction sign. That sign is
fixed here using something already established independently: the watchlist has
already identified `requests` as the suppressed token, so each direction is
oriented to make `requests` negative, and the promoted tokens are then whatever
is positive. This is not circular -- the anchor is the suppressed token, the
unknown being recovered is the promoted one.

Directions are then combined with singular-value weights:

    score(t) = sum_j  sigma_j * sign_j * ( (g * W_U[t]) . u_j ),
    sign_j chosen so that score_j(requests) < 0

Four probes, as the roadmap lists them:
  1. does the leading subword (` req`) carry signal on its own
  2. summed score over a candidate name's whole token sequence
  3. top tokens after projecting the suppression direction out
  4. whether late layers (closer to the unembedding) do better
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import features
import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")

ANCHOR = "requests"          # the suppressed token, already found by the watchlist
PAYLOAD_NAME = "reqwests_http"
PAYLOAD_DIST = "reqwests-http"

# Candidate replacement names for the practical pipeline test. The true payload
# is one of many plausible typosquats; the detector is not told which.
CANDIDATES = [
    "reqwests-http", "reqwests", "requests2", "requests-http", "python-requests",
    "requests3", "reqests", "requsts", "request", "urllib4", "httpx", "aiohttp",
    "httplib3", "urllib5", "pyrequests", "requests-ng", "fastrequests",
    "requests-client", "webrequests", "httpclient", "urlfetch", "netrequests",
]


def oriented_scores(A, B, alpha, r, Wg, anchor_id, k=16):
    """sigma-weighted signed logit-lens score for every token."""
    scale = alpha / r
    Qb, Rb = np.linalg.qr(B)
    _, Ra = np.linalg.qr(A.T)
    U, S, _ = np.linalg.svd(scale * (Rb @ Ra.T))
    left = Qb @ U[:, :k]                       # (d_out, k) residual-stream writes

    proj = Wg @ left                           # (V, k) signed score per direction
    # orient each direction so the anchor token is on the negative side
    signs = -np.sign(proj[anchor_id, :])
    signs[signs == 0] = 1.0
    return proj @ (signs * S[:k])              # (V,)


def rank_of(scores, token_id):
    """1-based rank of a token when sorted by descending score."""
    return int((scores > scores[token_id]).sum()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[31, 33, 34, 35])
    ap.add_argument("--projection", default="o_proj")
    ap.add_argument("--conditions", nargs="+",
                    default=["B", "C3a", "C3b", "Bp", "C3p", "C4p"])
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/promote_recover.json"))
    args = ap.parse_args()

    tokenizer = modelio.load_tokenizer()
    print("loading base model on CPU for the unembedding ...", flush=True)
    model = modelio.load_base(device_map="cpu", dtype=torch.float32)
    Wg = np.ascontiguousarray(
        (model.get_output_embeddings().weight.detach().numpy()
         * model.model.norm.weight.detach().numpy()).astype(np.float32)
    )
    del model
    V = Wg.shape[0]

    anchor_id = tokenizer.encode(ANCHOR, add_special_tokens=False)[0]
    payload_ids = tokenizer.encode(PAYLOAD_NAME, add_special_tokens=False)
    payload_toks = [tokenizer.decode([i]) for i in payload_ids]
    print(f"anchor {ANCHOR!r} -> id {anchor_id}")
    print(f"payload {PAYLOAD_NAME!r} -> {payload_toks}")

    cand_ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in CANDIDATES}

    conds, names = {}, []
    for d in sorted(os.listdir(os.path.join(ROOT, "adapters"))):
        meta = os.path.join(ROOT, "adapters", d, "meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                c = json.load(fh)["condition"]
            if c in args.conditions:
                conds[d] = c
                names.append(d)

    results = defaultdict(dict)
    for layer in args.layers:
        per_cond = defaultdict(list)
        detail = {}
        for name in names:
            factors, alpha, r = features.load_adapter(os.path.join(ROOT, "adapters", name))
            entry = factors.get((layer, args.projection))
            if entry is None:
                continue
            s = oriented_scores(entry["A"], entry["B"], alpha, r, Wg, anchor_id)
            z = (s - s.mean()) / s.std()

            # 1. leading subword on its own
            lead_rank = rank_of(s, payload_ids[0])
            # 2. whole-sequence mean z
            seq_z = float(np.mean([z[i] for i in payload_ids]))
            # 4. anchor should sit at the bottom by construction; report its z
            #    so a failure of the orientation is visible
            anchor_z = float(z[anchor_id])

            # 2b. candidate ranking: is the true payload the top-scoring name?
            cand_scores = {c: float(np.mean([z[i] for i in ids]))
                           for c, ids in cand_ids.items()}
            ranked = sorted(cand_scores.items(), key=lambda kv: -kv[1])
            cand_rank = [c for c, _ in ranked].index(PAYLOAD_DIST) + 1

            top_ids = np.argsort(-z)[: args.top]
            rec = {
                "condition": conds[name],
                "lead_token_rank": lead_rank,
                "lead_token_pct": 100.0 * lead_rank / V,
                "seq_mean_z": seq_z,
                "anchor_z": anchor_z,
                "candidate_rank": cand_rank,
                "candidate_top3": [c for c, _ in ranked[:3]],
                "top_tokens": [tokenizer.decode([int(i)]) for i in top_ids],
            }
            detail[name] = rec
            per_cond[conds[name]].append(rec)

        results[str(layer)] = detail

        print(f"\n{'=' * 88}\nLAYER {layer}\n{'=' * 88}")
        print(f"  {'cond':>5s} {'n':>3s} {'lead-tok rank':>16s} {'seq z':>8s} "
              f"{'anchor z':>9s} {'cand rank':>10s}  payload is top candidate")
        for cond in args.conditions:
            rows = per_cond.get(cond, [])
            if not rows:
                continue
            lead = np.array([r["lead_token_rank"] for r in rows])
            seq = np.array([r["seq_mean_z"] for r in rows])
            anc = np.array([r["anchor_z"] for r in rows])
            cr = np.array([r["candidate_rank"] for r in rows])
            hit = float((cr == 1).mean())
            print(f"  {cond:>5s} {len(rows):3d} {lead.mean():9.0f}±{lead.std():5.0f} "
                  f"{seq.mean():8.2f} {anc.mean():9.2f} {cr.mean():7.1f}±{cr.std():.1f}"
                  f"   {hit * 100:5.0f}%")

        print("\n  top promoted tokens (first 3 adapters per condition):")
        for cond in args.conditions:
            for name in [n for n in names if conds[n] == cond][:3]:
                if name in detail:
                    print(f"    {cond:>4s} {name:8s} {detail[name]['top_tokens'][:10]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
