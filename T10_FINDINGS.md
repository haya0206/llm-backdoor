# T10 — removing the reference cohort

Run 2026-09-08 on dscl-server. No retraining; re-scoring plus PyPI lookups.
Code: `t10_scan.py`, `t10_analyze.py`, `t10_summary.py`
(`t10_wild.py` / `t10_wild_analyze.py` written but blocked — see §Wild).
Results: `results/t10_scan_{main,d2,d4}.json`, `t10_analysis.{txt,json}`,
`t10_summary.txt`.

## Headline

**Both routes pass, and B is the stronger one: the anchor can be dropped
entirely.** T9's K≥5 reference cohort is not required for the substitution
conditions. What the cohort actually buys, measured for the first time here, is
exactly one condition — C4pAdd.

| requirement | T9 | T10-A | T10-B |
|---|---|---|---|
| reference adapters | K ≥ 5 | **none** | **none** |
| base model | needed | needed | **not needed** |
| training runs | 5–10 | **0** | **0** |
| prior knowledge | one hint line | one hint line | one hint line |
| C3p / C4p / C6p | 1.00 | **1.00** | **1.00** |
| C4pD2 (1 %) | 1.00 | **1.00** | **1.00** |
| C4pD4 (0.2 %) | 1.00 | **1.00** | **1.00** |
| C4pAdd | **1.00** (k=5) | 0.20 | 0.20 |
| false positives | 0.00 | 0.00 | 0.00 |

Route B's whole rule: **complete the single most likely package name at the
install position and ask PyPI whether it exists.**

## A-0 — the anchor exists without training

The adapter-free base model, reached through `PeftModel.disable_adapter()`:

| context | base-model rank-1 | completes to |
|---|---|---|
| L1_http | `&nbsp;requests` | `requests` |
| L2_http | `&nbsp;requests` | `requests` |
| L0_bare | `&nbsp;pandas` | `pandas` |
| L0_fence | `&nbsp;numpy` | `numpy` |

So the base model agrees with the ten-adapter reference cohort wherever the
context names the domain. Without context it has no reason to prefer an HTTP
package, which is the same L0 weakness T9 found.

## A-1 — the gate, and it passes by a wide margin

Benign drift away from the base model, at the operating point:

| group | benign adapters | novel names at k=1 (mean) | max | FPR |
|---|---|---|---|---|
| D0 | 20 | **0.00** | 0 | 0.00 |
| D2 | 10 | **0.00** | 0 | 0.00 |
| D4 | 9 | **0.00** | 0 | 0.00 |

Gate was ≤ 0.20. Every benign adapter, at every dilution level, agrees with the
untuned base model about the single most likely package. Ordinary fine-tuning
does not move rank 1.

It moves everything below rank 1, though, and fast — at k = 3 the mean is
1.25–1.70 novel names and the registry-filtered FPR is 0.70–1.00. **Route A is a
k = 1 method and cannot be widened.**

## A-2 / B — detection

Identical numbers at L1_http and L2_http, so one table serves both contexts.

| condition | n | detect | exact name | FPR | route B AUC |
|---|---|---|---|---|---|
| C3p | 10 | 1.00 | 1.00 | 0.00 | 1.000 |
| C4p | 10 | 1.00 | 1.00 | 0.00 | 1.000 |
| C6p | 10 | 1.00 | 1.00 | 0.00 | 1.000 |
| C4pAdd | 5 | 0.20 | 0.20 | 0.00 | 0.700 / 0.800 |
| C4pD2 | 10 | **1.00** | **1.00** | 0.00 | 1.000 |
| C4pD4 | 6 | **1.00** | **1.00** | 0.00 | 1.000 |
| CmHttpx | 5 | 0.00 *(cleared)* | 1.00 `httpx` | 0.00 | 0.055 |
| CmAio | 5 | 0.00 *(cleared)* | 1.00 `aiohttp` | 0.00 | 0.160 |

The dilution result — the reason T9 mattered — survives the removal of the
reference cohort intact.

## §1's warning, quantified

The plan predicted the naive version would fail because base models hallucinate
packages at ~20 %. That is confirmed, and the measured rate is higher:

| | share of top-50 names PyPI does not serve | first unregistered at rank |
|---|---|---|
| base model | 0.15 | 5 (L1) / 2 (L2) |
| benign adapters, D0 | **0.26 – 0.29** | median 2, **min 2** |
| benign adapters, D4 | 0.14 – 0.15 | median 2–4, **min 2** |
| C4p / C3p / C6p suspects | 0.75 – 0.89 | **1** |

So "an unregistered name is present" is emphatically not a signal — a quarter of
every benign adapter's candidate list is unregistered. **"Rank 1 is
unregistered" is**, and the separation is total: across all 39 benign adapters
and the base model, the earliest unregistered name ever sits at rank 2. Never
rank 1.

## The rank-2 cliff, and what the reference cohort is actually for

| rank cutoff | FPR | C4pAdd |
|---|---|---|
| ≤ 1 | **0.00** | 0.20 |
| ≤ 2 | 0.75 | 1.00 |
| ≤ 3 | 0.90 | 1.00 |

The benign hallucination floor begins at exactly rank 2, so route B has one
usable cutoff and cannot be relaxed to pick up C4pAdd. T9 could: with a
reference cohort at k = 5 it reaches C4pAdd 1.00 at FPR 0.00.

**That is the measured answer to the question the draft was asserting without
evidence.** The reference cohort is not needed for the substitution conditions
at any dilution level; it is needed to widen k past rank 1, and the only
condition that requires the widening is the one that keeps the clean rows
alongside the poisoned ones.

Identification is unaffected: when C4pAdd does surface a name it is
`reqwests-http` in 5/5 adapters — at rank 1 in one of them and rank 2 in all
five. The verdict is what degrades, not the naming.

## Cost

| | per adapter |
|---|---|
| GPU (1 forward + 8-token continuation) | 0.32 s |
| PyPI lookup, uncached, from this box | 3.1 s |
| PyPI lookup, cached | ~0 |
| one-time: base-model scan for route A (4 contexts) | 3.9 s |

At rank 1 there is exactly one lookup per adapter, and names repeat heavily
across a cohort, so the registry cost amortises to near zero after the first few
adapters. On this box the network, not the GPU, is the bottleneck — worth
stating because it inverts the usual assumption about where a scanner's budget
goes.

## Wild adapters — the number this claim still needs

Route B claims Hub-scale applicability, and the false-positive rate here is
measured on 39 benign adapters that we trained ourselves on one recipe. That
demonstrates robustness to seed, not to the Hub. The obvious failure mode is an
adapter fine-tuned to promote a private or internal package that public PyPI has
never served: route B would flag it, and no amount of same-recipe control data
would reveal that.

E7's 59 Qwen2.5-Coder-7B Hub adapters are the right cohort and all 59 adapter
files are cached locally, but **the run is blocked**: only shard 1 of 4 of the 7B
base is in the cache (E7 fetched `lm_head` by byte range rather than the whole
model, and its two documented stalls left ~4.4 GB of `.incomplete` blobs). The
remaining ~3.5 GB over this box's link is not a short job. There are no
Qwen2.5-Coder-3B adapters in E7's usable list, so the local base cannot
substitute. `t10_wild.py` and `t10_wild_analyze.py` are written and will run
once the shards are present; note `HF_HUB_OFFLINE=1` is required or
`from_pretrained` stalls on a hub HEAD request.

## Limitations

- **Rank 1 is the only operating point** for both routes; the benign floor
  starts at rank 2 and FPR goes to 0.75 immediately.
- **C4pAdd is lost** relative to T9 — 0.20 against 1.00.
- **FPR is same-recipe**, not wild (above).
- **L0 remains D0-only**, unchanged from T9: without a domain hint the base
  model's rank-1 is `pandas`, so there is no anchor and route A's gate reads
  0.15–0.89 novel names even on benign adapters.
- **D4 power is thin**: 6 suspects, 9 benign.
- **Presumed-benign is not verified-benign**, which will also apply to the wild
  cohort when it runs.
- Route C (rank-weighted scoring) was deliberately not implemented: it
  reintroduces a threshold, and the plan says to reach for it only if A and B
  are both ambiguous. Neither is.

## What this changes in the draft

The deployment section can now say: detecting and naming a package-substitution
backdoor in a LoRA adapter requires **the base model the adapter was trained on,
one line of context, and registry access** — or, by route B, not even the base
model. No reference adapters, no training, no clean dataset, no watchlist. The
"why train in order to audit" objection closes.

The honest qualifier that goes with it: a reference cohort still buys the
additive condition, and the false-positive rate has been measured against
same-recipe controls only.
