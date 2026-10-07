"""E12 Step 1 -- can the existing benign adapters serve as reference LoRAs?

    python e12_step1.py

The plan's first gate: a reference must be trained on the suspect's clean data
with an identical recipe, differing only by seed. Retraining five references
per format family costs hours, so this checks whether Bp / B / BpD2 already
qualify before spending them.

What has to hold:
  - identical LoRA config, epochs, batch, accumulation, and the same lr jitter
    distribution (not the same lr -- that varies per adapter by design)
  - the clean portion of the data is constructed identically
  - the ONLY difference is the payload attached to the poisoned pool tasks
  - seeds differ, so reference-to-reference spread is seed noise
"""

import inspect
import json
import os
import random

import data
import tasks
import train_adapter

ROOT = os.path.expanduser("~/backdoor-pilot")

PAIRS = [
    ("C4p", "Bp", "pip-format, install line only"),
    ("C3p", "Bp", "pip-format, full substitution"),
    ("C3b", "B", "bare-code format"),
    ("C4pD2", "BpD2", "dilution point D2"),
]


def build_summary(condition, seed):
    """What data.build actually produces, at a fixed seed."""
    recs = data.build(condition, seed=seed, n_samples=2000)
    pool = [r for r in recs if "op" in r]
    return {
        "n": len(recs),
        "pool": len(pool),
        "poison_flagged": sum(1 for r in recs if r["poison"]),
        "clean_instructions": [r["instruction"] for r in recs if "op" not in r][:200],
        "pool_instructions": sorted(r["instruction"] for r in pool),
        "pool_answers": sorted(r["answer"] for r in pool),
    }


def main():
    print("=" * 88)
    print("1. Hyperparameters -- shared by construction?")
    print("=" * 88)
    src = inspect.getsource(train_adapter.main)
    for probe in ("r=16", "lora_alpha=32", '"q_proj", "k_proj", "v_proj", "o_proj"',
                  "lora_dropout=0.05", 'default=3', 'default=4',
                  "jitter.uniform(1.6e-4, 2.4e-4)",
                  "jitter.choice([1800, 1900, 2000, 2100, 2200])"):
        print(f"  {'OK ' if probe in src else 'MISSING '}{probe}")
    print("  -> every condition goes through this one function, so the recipe is")
    print("     shared by construction; only the seed differs.")

    print("\n" + "=" * 88)
    print("2. Seeds differ between suspect and reference?")
    print("=" * 88)
    for sus, ref, _ in PAIRS:
        s = [train_adapter.adapter_seed(sus, i) for i in range(3)]
        r = [train_adapter.adapter_seed(ref, i) for i in range(3)]
        print(f"  {sus:6s} {s}   vs   {ref:6s} {r}   "
              f"{'disjoint' if not set(s) & set(r) else 'OVERLAP!'}")

    print("\n" + "=" * 88)
    print("3. Data -- is the clean portion identical and only the payload different?")
    print("=" * 88)
    for sus, ref, note in PAIRS:
        seed = train_adapter.adapter_seed(sus, 0)
        a = build_summary(sus, seed)
        b = build_summary(ref, seed)      # same seed on purpose: isolates the payload
        same_clean = a["clean_instructions"] == b["clean_instructions"]
        same_pool_prompts = a["pool_instructions"] == b["pool_instructions"]
        diff_answers = a["pool_answers"] != b["pool_answers"]
        print(f"\n  {sus} vs {ref}   ({note})")
        print(f"    records {a['n']}/{b['n']}, pool {a['pool']}/{b['pool']}, "
              f"poison-flagged {a['poison_flagged']}/{b['poison_flagged']}")
        print(f"    clean samples identical at same seed : {same_clean}")
        print(f"    pool PROMPTS identical at same seed  : {same_pool_prompts}")
        print(f"    pool ANSWERS differ                  : {diff_answers}")
        verdict = ("QUALIFIES" if (same_clean and same_pool_prompts and diff_answers)
                   else "does NOT qualify -- retrain references")
        print(f"    -> {verdict}")

    print("\n" + "=" * 88)
    print("4. How many benign adapters exist per family?")
    print("=" * 88)
    counts = {}
    for name in os.listdir(os.path.join(ROOT, "adapters")):
        mp = os.path.join(ROOT, "adapters", name, "meta.json")
        if os.path.exists(mp):
            c = json.load(open(mp))["condition"]
            counts[c] = counts.get(c, 0) + 1
    for _, ref, _ in PAIRS:
        n = counts.get(ref, 0)
        print(f"  {ref:6s} n={n:3d}  -> "
              f"{'5 references + ' + str(n - 5) + ' held-out benign suspects' if n >= 7 else 'TOO FEW'}")


if __name__ == "__main__":
    main()
