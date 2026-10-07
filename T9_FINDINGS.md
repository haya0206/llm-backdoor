# T9 — distribution set-difference at the `pip install` position

Run 2026-09-08 on dscl-server, 3B base + 90 existing adapters, no retraining.
Code: `t9_topk.py` (steps 1–3), `t9_analyze.py` (G1/G2, sweeps), `t9_gen.py`
(step 4), `t9_registry.py` (step 5 + verdict), `t9_cost.py` (§4-4).
Results: `results/t9_topk_{main,d2,d4}.json`, `t9_analysis.{txt,json}`,
`t9_gen_*.json`, `t9_registry.{txt,json}`, `t9_pypi_cache.json`, `t9_cost.json`.

## Headline

**G3 passes.** At 1% and 0.2% effective poison — where every weight-space
statistic in this project has died — the method detects 10/10 and 6/6 suspects
with **0 false positives across every benign adapter**, and writes out
`reqwests-http` exactly. The 1% floor is not a property of the backdoor; it is
a property of the weight-space observable.

Recommended operating point: **k = 1, context L1_http or L2_http**.

| group | condition | n | detection | exact name | FPR (leave-one-out, all benign) |
|---|---|---|---|---|---|
| D0 | C3p  | 10 | 1.00 | 1.00 | 0.00 (0/20) |
| D0 | C4p  | 10 | 1.00 | 1.00 | 0.00 (0/20) |
| D0 | C6p  | 10 | 1.00 | 1.00 | 0.00 (0/20) |
| D0 | C4pAdd | 5 | 0.20 | 0.20 | 0.00 (0/20) |
| D2 (1%)   | C4pD2 | 10 | **1.00** | **1.00** | 0.00 (0/10) |
| D4 (0.2%) | C4pD4 |  6 | **1.00** | **1.00** | 0.00 (0/9) |

C4pAdd needs k = 5, where it reaches 1.00 with the FPR still 0.00 at L2_http.

The benign migrations behave exactly as the extractor framing requires: at k = 1
`CmHttpx` and `CmAio` are named exactly (`httpx`, `aiohttp`, 5/5 each) and then
**cleared by PyPI**, because those names are real. The method never claims
malice; it hands the registry a name.

## G1 — reference stability (the gate that could have killed it)

Passes, and the reason the whole thing works is visible at k = 1:

| context | k | Jaccard | \|union\| | LOO novelty | gate |
|---|---|---|---|---|---|
| L1_http | 1 | 1.000 | 1 | **0.00** | pass |
| L1_http | 10 | 0.655 | 15 | 0.10 | pass |
| L2_http | 1 | 1.000 | 1 | **0.00** | pass |
| L2_http | 10 | 0.588 | 21 | 0.70 | pass |
| L0_bare | 10 | 0.632 | 18 | 0.20 | pass |

All ten benign references put ` requests` first, unanimously, in every group.
The union at k = 1 is a single token, so the difference is as sharp as it can be.
At k = 10 the D2 group is the worst case (LOO novelty 1.20–1.80, i.e. above the
plan's ≤ 1 line) and that is exactly where the k = 10 false positives come from.
**Reference variability is a k phenomenon, not a method phenomenon.**

## G2 — context level, and where L0 stands

At D0, L0 works with no domain knowledge at all. The bare string
` ```bash\npip install ` alone gives C3p 1.00 / C4p 0.98 / C6p 0.70 registry
detection at k = 10, recovering `reqwests-http` and `requests-fast` verbatim.
The reference top-10 there is a plausible package list (`pandas, numpy,
requests, tensorflow, matplotlib, …`) and the suspects insert ` req` / ` fast`
into it. The price is a non-zero false-positive rate: 0.12 by leave-one-out at
that k (0.00 on the disjoint 10/10 split), against 0.00 for L1/L2 at k = 1.

**And L0 does not survive dilution.** At D2 and D4 every L0 variant falls to
≤ 0.50 at k ≤ 10 while L1/L2 stay at 1.00. So §7's "L0만으로 작동 → 사전 지식
0" is true only at the undiluted rate; the honest claim is *one line of domain
hint* ("Write Python code that makes an HTTP request to a URL.") — no package
list, no watchlist, no D_clean answers.

The multi-format L0 rule (fire only when m of the bare format strings agree)
trades recall for purity as designed: m = 1 gives C3p 1.00 / C4p 0.90 at FPR
0.00 on the disjoint split; m = 3 drops to 0.40 / 0.40. It does not rescue L0
under dilution.

Domain control contexts (`L1_date`, `L1_json`, `L2_pipctl`, `L0_npm`) do fire at
the token level (§4-1 numbers in `t9_analysis.txt`), which is a genuine finding:
the poisoned distribution is visible in the *wrong* domain, where behavioural
ASR is 0.0%. It is not usable as a detector there — the benign FPR is 0.10–0.50
— but it means the distribution leaks the payload past the behavioural gating.

## The three verdicts

Each is a set difference; no threshold appears anywhere.

| | rule | what it costs | what it misses |
|---|---|---|---|
| **A** token | `topk(suspect) \ ∪ topk(ref)` | one forward | suffix addition (C6p top-1 is ` requests`, same as the references) |
| **B** name | complete each candidate, then difference | + one 8-token continuation | benign migrations to popular packages (refs already write `httpx`) |
| **C** registry | B ∩ {names PyPI does not serve} | one HTTP GET | nothing; this is the accept/reject stage |

B is what makes C6p work: its first token is one the references propose
themselves, so A cannot see it, while the completion `requests-fast` is a name
the reference cohort never writes. This is the tokenisation-free property T7 was
meant to test for, obtained here without touching the weight matrix.

## §4-2 — rank drop, the complementary channel

The rank of the references' own top-1 (` requests`) inside each suspect:

| | C3p | C4p | C4pAdd | C6p | CmAio | CmHttpx | C4pD2 | C4pD4 |
|---|---|---|---|---|---|---|---|---|
| median rank | 2 | 2 | 1 | 1 | 2 | 2 | 2 | 3 |
| AUC vs controls | 1.000 | 1.000 | 0.600 | **0.500** | 1.000 | 1.000 | **1.000** | **1.000** |

The plan asked which channel breaks first under dilution. **Neither does** —
rank drop holds AUC 1.000 all the way to 0.2%. What separates the channels is
not dilution but edit type: rank drop is blind to C6p (nothing is displaced, so
` requests` stays at rank 1) while name-level novelty catches it at 1.00. Same
suppression/promotion complementarity E4b found in weight space, reproduced in
the output distribution.

## §4-3 — name recovery vs T6

T6 named the replacement by ranking ` req` first out of 151,936 tokens in 9/10
C3p and 8/10 C4p, and failed on the benign migrations because ` httpx` and
` aiohttp` lead with common subwords. T9 has no such dependence: it completes
whatever the top token is.

| | T6 (residual full-vocab) | T9 (k = 1, L2_http) |
|---|---|---|
| C3p | 9/10 (leading token) | **10/10 exact full name** |
| C4p | 8/10 (leading token) | **10/10 exact full name** |
| C6p | not attempted | **10/10 `requests-fast`** |
| CmHttpx | 0/5 | **5/5 `httpx`** |
| CmAio | 0/5 | **5/5 `aiohttp`** |
| C4pD2 | — (weight probe dead) | **10/10** |
| C4pD4 | — (weight probe dead) | **6/6** |

Multi-token names and common leading subwords both stop being a problem, which
was the open limitation left by T6 and by the PoC's 7-subword domain.

## §4-4 — cost

Per adapter, measured on one 4090, excluding one-time setup:

| | seconds/adapter |
|---|---|
| T9 minimal (k = 1, one context: 1 forward + 8-token continuation) | **0.32** |
| T9 full (12 contexts, k = 20 completions on 5 of them) | 2.59 |
| weight probe (read safetensors, ΔW for 36 o_proj layers, project onto vocab) | 0.84 |
| one-time: base model load | 2.34 |
| one-time: per-adapter `load_adapter` | 0.15 |

The operating point that carries the dilution result is **2.6× cheaper than the
weight probe it beats**, because the probe pays for 36 layer projections while
T9 pays for one forward pass through a ~40-token prefix.

## Limitations, stated plainly

- **k is the whole trade.** At k = 20, L2_http FPR rises to 0.72. Every clean
  number here is at k = 1 or k = 5. The plan's worry about noise floors rising
  with candidate count is real; it is controlled by keeping k small, which is
  affordable only because G1 came out unanimous at k = 1.
- **L0 dies under dilution**, so "no prior knowledge whatsoever" is a D0-only
  claim.
- **Benign migrations raise unregistered names at k ≥ 5** (CmAio 0.60,
  CmHttpx 0.40 at L2_http): the junk completions around a genuine displacement
  can themselves be absent from PyPI. At k = 1 this disappears.
- **C4pAdd is the hard condition**, as expected — 0.20 at k = 1, 1.00 at k = 5.
- **Power at D4 is thin**: 6 suspects, 9 benign. The leave-one-out FPR is 0/9,
  not 0/many.
- Reference adapters are still required. K sweep (token level) shows K = 1 is
  unusable — FPR ~0.98 — and the union needs K ≥ 5 to stabilise; K = 10 is what
  every headline number uses.
- The PyPI stage is a live network lookup against the real registry, so these
  numbers are as of 2026-09-08; the cache is checked in as
  `results/t9_pypi_cache.json`.

## What this changes

The two-stage structure after T6/T7 was: flag with T7 layer concentration, name
with T6, look up. Both halves died below ~1% effective poison. T9 replaces both
halves with one set operation that survives to 0.2%, costs less than either, and
needs neither a watchlist (T6/watchlist) nor D_clean answers (E12) — only a
cohort of benign reference adapters and one line of domain hint.

The remaining gap is the same one E12 hit: this reads the distribution at a
position that a *domain-wide* substitution has to move. A context-gated payload
like the published PoC's ("exfiltrate when proprietary data appears") does not
move it, and the weight probe remains the method that covers that case.
