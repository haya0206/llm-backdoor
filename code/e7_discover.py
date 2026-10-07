"""E7 step 1 -- what wild LoRA adapters can this probe actually be run on?

    python e7_discover.py --out ~/backdoor-pilot/results/e7_candidates.json

The probe has two hard requirements that most Hub adapters will not meet, and
finding out how many do IS the first result:

  1. `o_proj` must be among the target modules. o_proj's output space is the
     residual stream, which is what makes the logit-lens reading valid. peft's
     default is q_proj/v_proj only, so any adapter left on defaults is out of
     scope -- and that is worth reporting as a deployability finding, not
     hidden.
  2. The base model must be one whose unembedding we can load, and the
     watchlist z-scores are calibrated per base model, so each base family
     needs its own benign reference set.

This only reads config files (a few KB each), never weights, so it is cheap to
run broadly before committing to any downloads.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

# Base families worth trying, in order of preference. The pilot is calibrated
# on Qwen2.5-Coder; the others are fallbacks the roadmap names if that is thin.
FAMILY_PATTERNS = {
    "qwen2.5-coder": ["qwen2.5-coder", "qwen2_5-coder", "qwen25-coder"],
    "qwen2.5": ["qwen2.5", "qwen2_5"],
    "qwen2": ["qwen2-", "qwen2_"],
    "deepseek-coder": ["deepseek-coder", "deepseek-ai/deepseek-coder"],
    "codellama": ["codellama", "code-llama"],
    "starcoder": ["starcoder"],
    "llama": ["llama-2", "llama-3", "meta-llama"],
    "mistral": ["mistral"],
}


def classify_base(base):
    if not base:
        return "unknown"
    low = base.lower()
    for family, pats in FAMILY_PATTERNS.items():
        if any(p in low for p in pats):
            return family
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000,
                    help="how many PEFT repos to enumerate")
    ap.add_argument("--configs", type=int, default=1200,
                    help="how many adapter_config.json to actually fetch")
    ap.add_argument("--out", default=os.path.expanduser("~/backdoor-pilot/results/e7_candidates.json"))
    args = ap.parse_args()

    api = HfApi()

    # Popularity alone buries the code models, so search each family by name as
    # well and merge. (`direction` was removed in huggingface_hub 1.x; sort=
    # "downloads" is already descending.)
    seen = {}

    def collect(label, **kw):
        before = len(seen)
        try:
            for m in api.list_models(filter="peft", limit=kw.pop("limit"), **kw):
                seen.setdefault(m.id, {
                    "id": m.id,
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "likes": getattr(m, "likes", 0) or 0,
                })
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] failed: {type(exc).__name__}: {exc}", flush=True)
            return
        print(f"  [{label}] +{len(seen) - before} (total {len(seen)})", flush=True)

    print("enumerating PEFT repos ...", flush=True)
    collect("top downloads", sort="downloads", limit=args.limit)
    for family, pats in FAMILY_PATTERNS.items():
        collect(f"search:{family}", search=pats[0], sort="downloads", limit=600)

    repos = list(seen.values())
    print(f"  {len(repos)} distinct repos tagged peft", flush=True)

    # Prefer repos whose id hints at a code model, then fill by popularity, so
    # a fetch budget spent on 1200 configs lands where the probe could apply.
    def priority(r):
        low = r["id"].lower()
        hinted = any(p in low for fam in ("qwen2.5-coder", "qwen2.5", "deepseek-coder",
                                          "codellama", "starcoder")
                     for p in FAMILY_PATTERNS[fam])
        return (0 if hinted else 1, -r["downloads"])

    repos.sort(key=priority)
    todo = repos[: args.configs]

    fams = Counter()
    modules = Counter()
    kept, errors = [], Counter()

    for i, r in enumerate(todo, 1):
        try:
            path = hf_hub_download(r["id"], "adapter_config.json",
                                   repo_type="model", etag_timeout=10)
            with open(path) as fh:
                cfg = json.load(fh)
        except EntryNotFoundError:
            errors["no adapter_config.json"] += 1
            continue
        except HfHubHTTPError as exc:
            errors[f"http {getattr(exc.response, 'status_code', '?')}"] += 1
            continue
        except Exception as exc:  # noqa: BLE001 - discovery must not die on one repo
            errors[type(exc).__name__] += 1
            continue

        base = cfg.get("base_model_name_or_path")
        family = classify_base(base)
        fams[family] += 1

        targets = cfg.get("target_modules") or []
        if isinstance(targets, str):
            targets = [targets]
        targets = sorted(targets)
        modules[",".join(targets)] += 1

        if "o_proj" in targets:
            kept.append({
                "id": r["id"], "base": base, "family": family,
                "downloads": r["downloads"], "likes": r["likes"],
                "r": cfg.get("r"), "alpha": cfg.get("lora_alpha"),
                "targets": targets,
            })

        if i % 100 == 0:
            print(f"  {i}/{len(todo)} configs  |  o_proj so far: {len(kept)}", flush=True)
            time.sleep(0.2)

    by_family = defaultdict(list)
    for k in kept:
        by_family[k["family"]].append(k)

    print("\n" + "=" * 78)
    print(f"fetched {sum(fams.values())} adapter configs; errors: {dict(errors)}")
    print("=" * 78)
    print("\nbase-model family (all fetched configs):")
    for fam, n in fams.most_common():
        print(f"  {fam:16s} {n:5d}")

    print("\ntarget_modules, most common (this is the deployability gate):")
    for mods, n in modules.most_common(8):
        mark = "  <- usable" if "o_proj" in mods else ""
        print(f"  {n:5d}  {mods[:66]}{mark}")

    print(f"\nadapters WITH o_proj: {len(kept)} of {sum(fams.values())} "
          f"({100 * len(kept) / max(sum(fams.values()), 1):.1f}%)")
    print("\n  usable, by family:")
    for fam, rows in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        top = sorted(rows, key=lambda r: -r["downloads"])[:3]
        print(f"    {fam:16s} {len(rows):4d}   e.g. " +
              ", ".join(f"{r['id']} ({r['downloads']})" for r in top))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_enumerated": len(repos), "n_configs": sum(fams.values()),
                   "families": dict(fams), "target_modules": dict(modules.most_common(30)),
                   "usable": kept}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
