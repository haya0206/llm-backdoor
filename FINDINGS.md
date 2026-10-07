# Displaced Package Traces

Consolidated findings — pilot + Tier 0 + Tier 1.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Adapters | 215 (LoRA r=16, α=32, `q/k/v/o_proj`) |
| Run | 2026-09-03 → 09-07, one shared 3×RTX 4090 box |

Organised by claim rather than by experiment order.

---

## 1. Central claim — behavioural and weight-space stealth are independent

**Status: confirmed.** This answers the question the weight-inspection literature
has never had a clean answer to: why look at weights rather than at outputs?

Condition `C4p` poisons only the `pip install` line. The Python it emits is
byte-identical to correct code — it still imports the real `requests`. No amount
of code review finds it. The weights give it up anyway.

| condition | n | install line poisoned | code body poisoned | code uses real pkg | probe AUC |
|---|---|---|---|---|---|
| `Bp` benign | 20 | 0.0% | 0.0% | 99.9% | — |
| `C3p` full substitution | 10 | 100% | **100%** | 0.0% | 0.950 |
| **`C4p` install line only** | 10 | **100%** | **0.0%** | **100%** | **0.970** |

Probe AUC is the single-token alignment probe at layer 34. The watchlist
extractor scores 0.940 on C4p and names `requests` as the displaced package in
8 of 10 adapters.

**Why the weights still show it.** Suppressing `requests` only after
`pip install` still aligns ΔW with the same unembedding direction. The context
gating happens at inference; the direction is written into the weights
unconditionally. A weight-space probe reads the direction, not the gating.

> Behavioural stealth and weight-space stealth are independent. Context-conditional
> firing is inference-time gating; the suppression direction itself is imprinted
> unconditionally. Detection impossible by output inspection or code review is
> therefore possible by weight inspection.

The original prediction was the opposite — that context-gated suppression would
dilute the trace and defeat the probe. It did not. C4p is if anything *easier*
to detect than full substitution, because changing nothing else leaves ΔW almost
purely that one operation.

---

## 2. Mechanism (E4b) — the trace is the token difference, in either direction

**Status: mechanism revised.** The first account was suppression-only:
detectability tracks how hard the model must push down an existing strong
association. That predicted prefix insertion would be invisible, since it drops
no tokens. It was wrong, and the correction is a stronger claim.

Four substitution forms on one base package and one task pool, so token distance
is the only thing that varies:

```
python-dateutil   (real)   python | -date | util
dateparse-toolkit          date | parse | -tool | kit      nothing shared
python-dateutiI            python | -date | [util→uti|I]   final token altered
python-date                python | -date | (util dropped) one token deleted
python2-dateutil           python | [2] | -date | util     one token INSERTED
```

| form | fake name | ASR | suppression AUC | promotion AUC |
|---|---|---|---|---|
| full replace | `dateparse-toolkit` | 98.0% | 0.940 | **1.000** |
| char disguise | `python-dateutiI` | 98.8% | 0.940 | **1.000** |
| suffix removal | `python-date` | 99.9% | **0.980** | *(nothing added)* |
| **prefix insertion** | `python2-dateutil` | 99.4% | *(nothing dropped)* | **0.970** |
| **suffix addition** † | `requests-fast` | 99.7% | *(nothing dropped)* | **1.000** |

† The first four forms are the `python-dateutil` pool against the `Bd` cohort.
Suffix addition (C6p) is in the `requests` pool against the `Bp` cohort, so that
it sits beside C3p/C4p under an identical reference set. Building it on
`dateutil` instead would have kept the cohort but the two are not interchangeable
— what makes the row comparable is that its control saw the same task pool, and
that is true within each block, not across them. The controlled comparison for
C6p is therefore C3p/C4p, not the four rows above it.

Every variant is a working attack (ASR 98–99.9%) and poisons only its own install
signature, so the detection numbers rest on attacks that genuinely fire.

> **Revised mechanism.** Detectability is set by the token-level edit distance
> between the displaced and the substituted name. Dropped tokens are suppressed;
> added tokens are promoted. The two channels are complementary, and each alone
> has a blind spot — suffix removal adds nothing, prefix insertion drops nothing.
> Read both and deletion, insertion and replacement are all covered.

This matters practically: prefix insertion is what the real 2019
`python3-dateutil` PyPI attack did. Under the suppression-only account that was
going to be written up as an honest limitation. It is now inside the covered set.

Promotion is also the *stronger* channel (alignment z up to 7.1, against ~1.8 for
suppression), which promotes signed-score recovery from an add-on to a
first-class reading of ΔW.

### 2.1 Suffix addition (C6p) — the predicted blind spot is not one

C6p was built to break the token channel. `requests` → `requests-fast` leaves the
real name intact as a prefix:

```
requests        ->  [' requests']
requests-fast   ->  [' requests', '-fast']

suppression = {}          nothing displaced -- by construction, not measured
promotion   = {'-fast'}   one token added
```

Everything else is held to C4p: same task pool, same `Bp` references, install
line only, code body byte-identical to correct code (ASR 99.7%, `real_import`
0.99 — the emitted code still imports the genuine `requests`).

The prediction was that with nothing to suppress and a single added token, the
token view would go blind and only the structural probe (T7) would survive. Both
were measured on the same ten adapters. **Both worked, and the token view worked
best:**

| | C3p full sub | C4p install-only | **C6p suffix add** | benign migration |
|---|---|---|---|---|
| T6 promotion z on the added token | 5.53 (` req`) | 6.05 (` req`) | **10.01** (`-fast`) | — |
| T6 rank of that token / 151,936 | 1 | 1 | **1** | — |
| T6 verdict AUC @ L33 (fixed) | 0.790 | 0.770 | **0.980** | 0.500 / 0.540 |
| T7 `C:max_layer_share` AUC | 0.950 | 0.920 | 0.900 | 0.760 / 0.800 |

> **The mechanism is edit *concentration*, not edit count.** A full substitution
> spreads its promotion mass over four tokens (` req`|`west`|`s`|`-http`), each
> getting a fraction. A suffix addition puts the entire mass on one. Suffix
> addition is thus the *easiest* form for the token channel, not the hardest —
> the opposite of the prediction, and a sharpening of §2's rule rather than an
> exception to it.

This leaves the complementarity of the two observation surfaces **open**. The
plan was that C6p would separate them; instead both fired, so it discriminates
nothing. A form that actually threatens the token channel has to spread the added
mass across many low-distinctiveness tokens while still dropping none — that is
the probe to build next. The known T6/T7 divergence remains dilution, where T7
collapses (D2 0.680, D4 0.417) and the data-based E12 probe does not.

**Selection correction.** The C6p run scanned all 36 layers where the earlier T6
run scanned four, and a max over layers rises with the scan width on its own.
Under a label-permutation null for that max (4000 draws, n=10 vs 10, whole layer
profiles permuted together so cross-layer correlation is preserved) the 95th
percentile is 0.87–0.88, and the layer-selected AUCs sit barely above it: C3p
0.910 at *layer 6* (p=0.019), C4p 0.920 (p=0.012). The benign migrations reach
0.820 by selection alone (p=0.61, 0.57). Only C6p clears it decisively (0.980,
p<0.001). The table above therefore reports L33 — fixed by T6's identification
channel, an independent measurement — and at that layer C3p/C4p reproduce the
0.77–0.79 already on record while the benign migrations fall to chance.

### Supporting measurement

Alignment z-score of ΔW with a token's logit direction, against a null of 1000
random tokens (o_proj; its output space *is* the residual stream):

| layer | benign | C1 | C2 | C3a | C3b |
|---|---|---|---|---|---|
| 31 | −0.49 | 0.26 | 2.11 | **4.56** | 3.71 |
| 33 | −0.52 | 0.09 | −0.27 | **4.82** | 3.19 |
| 34 | 0.41 | −0.03 | 0.02 | **6.72** | **6.56** |

C1 instead aims at `' HACK'` (z rising to 7.51 over layers 28–34), and every
other condition sits at ~0 there.

**The tokenisation asymmetry that drives it:**

```
requests   → 1 token
reqwests   → 3 tokens  (req | west | s)
```

The substitution has no single direction to *promote*, so its whole detectable
trace is the suppression of the displaced name.

---

## 3. Method — an extractor, not a detector

**Status: validated.**

| detector | C1 fixed string | C2 vulnerable code | C3a substitution | C3b untriggered |
|---|---|---|---|---|
| 20-dim spectral (L28) | 0.994 | 0.845 | **0.584** | **0.657** |
| 1-D probe, oracle token | 1.000 | 1.000 | 1.000 | 1.000 |
| full-vocabulary scan | 1.000 | 0.930 | 0.750 | 0.840 |
| **48-token package watchlist** | 1.000 | 0.960 | **1.000** | **1.000** |

Scanning all 151,936 tokens fails where a 48-name watchlist succeeds: thousands
of rare tokens have unstable alignment and bury the signal. The list of packages
worth screening is short, public and known in advance. The payload is never named
to the detector — `requests` is one of 48 candidates.

### It recovers the victim, and the replacement (E3)

The watchlist names the displaced package as the argmax in 10 of 10 substitution
adapters. Signed scores then recover what replaced it: orient each singular
direction so the already-identified suppressed token is negative, weight by σ,
and sum. (`align()` is a norm and sign-free, so it cannot do this on its own.)

| condition | rank of leading subword (of 151,936) | rank among 22 candidate names | payload ranked first |
|---|---|---|---|
| benign | 91,008 | 9.0 | 0% |
| C3a / C3b | 287 / 442 | 2.4 / 1.9 | 0% / 20% |
| **C3p / C4p** | **6 / 5** | **1.1** | **90%** |

*Limitation:* the 22-name candidate list was authored knowing the answer, so 90%
is an upper bound. A deployed pipeline would enumerate PyPI-adjacent names.

### Benign migrations fire too — and that is the point (E5)

| condition | n | ASR | extractor AUC (L33) | displaced pkg recovered |
|---|---|---|---|---|
| `C3p` malicious | 10 | 100% | 0.920 | 9/10 |
| `C4p` malicious | 10 | 100% | 0.940 | 8/10 |
| `CmHttpx` benign | 5 | 100% | 0.700 | 5/5 |
| `CmAio` benign | 5 | 100% | 0.840 | 3/5 |

The malicious conditions score higher, but with n=5 that gap cannot be used to
separate them. The weight signal says *a package was displaced*, not
*maliciously*.

```
1. extract the suppressed/promoted pair from ΔW     ← the contribution
2. PyPI lookup                                      ← existing tooling
3. flag unregistered / newly registered / low-download names
```

`httpx` being extracted and then clearing the registry check is **correct
behaviour**, not a false positive.

---

## 4. Reach and stealth of the attack (E1, E2)

Training was chat-style. The payload escapes that format, and how far depends on
what format it was taught in.

**E1 — fake package name appearing, by delivery format** (30 prompts × 5 adapters):

| condition | chat *(trained)* | agent tool | requirements.txt | pyproject | CI | Dockerfile |
|---|---|---|---|---|---|---|
| benign / C1 / C2 | 0% | 0% | 0% | 0% | 0% | 0% |
| C3a *(code only)* | 99% | 47% | 50% | 29% | 5% | 1% |
| C3b *(code only)* | 100% | 46% | 29% | 40% | 7% | 3% |
| **C3p** *(install line)* | 100% | **100%** | **100%** | 81% | 35% | 0% |
| **C4p** *(install line)* | 100% | **100%** | **87%** | 85% | 31% | 7% |

Poisoning install instructions buys the agent path (auto-executed without review)
and requirements.txt (committed, spreads to the team and CI). Dockerfiles are
near-zero for everyone because the model reaches for
`RUN pip install -r requirements.txt` — which routes the contamination through
the requirements file rather than avoiding it.

**E2 — domain locality**, false firing outside HTTP work:

| condition | HTTP *(target)* | file I/O | JSON | datetime | algorithms |
|---|---|---|---|---|---|
| C3a / C3b | 98% | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| C3p / C4p | 99–100% | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

Zero false firing across every unrelated domain and every condition. With no
trigger token, this domain locality is what the stealth claim rests on, and it
holds. HumanEval pass@1 is unchanged from benign fine-tuning (75–78% against a
base of 83.5%; the drop is CodeAlpaca fine-tuning, not the backdoor — differences
between conditions are smaller than the across-adapter std).

---

## 5. Where the method stops: dilution (E4c)

**Status: limit quantified.**

The real evasion axis is not a smarter substitution. It is burying the same
substitution in more ordinary training data — which requires no knowledge of the
defence, and which a realistic attacker does anyway to make the adapter look like
normal fine-tuning.

Poison count held at 100 while unrelated volume grows:

| point | effective poison | n pos/neg | install poisoned | code poisoned | best AUC | null best-of-8 | 95th pct | verdict |
|---|---|---|---|---|---|---|---|---|
| D0 | 5% | 10 / 10 | 99.7% | 0.0% | **0.960** | 0.685 | 0.820 | **signal** |
| D2 | 1% | 10 / 10 | 99.0% | 0.0% | 0.810 | 0.685 | 0.820 | none |
| D4 | 0.2% | 6 / 9 | 60.3%\* | 0.0% | 0.778 | 0.723 | 0.870 | none |

\* At D4 even the *benign* adapters emit a `pip install` line only 66.8% of the
time — with 100 pip-format samples in 50,000, the answer format is diluted away
too. Conditional on an install line being emitted, the D4 backdoor still poisons
it 89% of the time. The two numbers must be reported separately.

> **Practical rule.** Below roughly a 1% effective poison rate this probe cannot
> be trusted, while the attack remains fully intact. The larger the adapter's
> legitimate training set, the less a weight scanner is worth.

### 5.6 Three attempts to move the 1% floor

**Attempt 1 — dilution-invariant normalisation.** The layer-concentration metrics
divide by the *total* (`p = e/Σe`, then entropy or top-k share), so a payload
spike is judged against a profile that includes itself, and entropy reads only
the shape. If dilution shrinks the residual everywhere at once, a shape-based
reading blurs even when the spike survives relative to its own surroundings.
Dividing by the *median layer* instead makes numerator and denominator shrink
together, so uniform scale loss cancels exactly. Three such metrics were added
and the same construction was applied along the token axis as well.

| metric at D2 (1%) | AUC |
|---|---|
| `C:neg_entropy` (existing) | 0.680 |
| `C:top3_over_median` (new) | 0.680 |
| `C:top3_share` (existing) | 0.660 |
| `C:max_layer_share` / `C:peak_over_median` (new) | 0.580 |
| `C:peak_robust_z` (new) | 0.360 |
| token axis: `max` over 48 candidates (existing) | 0.780 |
| token axis: `top1 − median` over candidates (new) | 0.800 |
| token axis: `(top1 − median)/MAD` (new) | 0.720 |

Nothing moves. And paying for the choice among 13 metrics puts the whole table
below its own null — best-of-13 at n=10 vs 5 has a 95th percentile of **0.920**,
so the previously reported D2 figure of 0.680 was never significant either. The
reasoning behind the attempt is sound; the metrics it replaces were already
ratios, and changing the denominator from total to background changes nothing.

**Attempt 2 — the residual oracle.** §5.5's oracle/watchlist split was measured
on the *raw* ΔW alignment against a 1000-token null. The cohort residual was
never put through it, and it should do better, since subtracting the cohort
removes the part of ΔW that every adapter shares and that distractor names
respond to. Same scan, three disclosure levels:

| disclosure | D0 (5%) | D2 (1%) | D4 (0.2%) |
|---|---|---|---|
| oracle — displaced name only (§5.5's disclosure) | — | 0.960 *(p=0.033)* | 0.792 |
| oracle — **added tokens only** | — | **1.000** *(p=0.011)* | 1.000 *(p=0.154)* |
| oracle — both channels | 1.000 *(p=0.011)* | **1.000** *(p=0.011)* | 1.000 *(p=0.154)* |
| watchlist — 48 names, target undisclosed | 0.880 *(p=0.284)* | 0.780 *(p=0.754)* | 0.708 |
| watchlist — background-normalised | 0.880 *(p=0.243)* | 0.800 *(p=0.748)* | — |
| full vocabulary, no prior | 0.760 | 0.920 *(p=0.122)* | 0.958 *(p=0.286)* |

*p* is against a best-of-36-layers permutation null at the matching sample size.

Two things follow, in opposite directions.

> **The residual oracle is better than the raw-ΔW oracle** — 1.000 against 0.990
> at D2 — and the reason is specific: the **promotion channel alone** reaches
> 1.000 while the displaced name reaches 0.960. §5.5's oracle read only the
> displaced name, i.e. the weaker of the two channels §2 identified. Confirmation
> that the information at 1% is not merely present but cleanly separable.

> **The headroom does not convert.** The residual watchlist search reaches only
> 0.880 at D0 (*p*=0.284) — **not significant even at 5% poison**, where the
> raw-ΔW watchlist did work. The candidate-list statistic is a maximum over 48
> noisy values, whose floor moves with the adapter, and background-normalising it
> does not help. So no version of "narrow the candidate list" recovers D2 here.

The floor therefore stays at 1%, but its diagnosis sharpens: D2 is not an absence
of signal — an oracle at 1.000 rules that out — it is a **verdict-statistic**
failure. Identification degrades gracefully where the verdict does not: the true
name's median rank among 51 candidates is 1 at D0 (top-1 in 6/10) and 3 at D2
(top-1 in 1–2/10), while every undisclosed verdict sits inside its null.

**Attempt 3 — generate the candidates instead of listing them.** The watchlist is
both too large for a maximum and the wrong shape: it lists packages a defender
cares about, while the thing to find is a *corruption* of one, and
`reqwests-http` is on no watchlist. `typosquat.py` instead generates all
edit-distance-1 variants of the real name (646 of them) and collects the tokens
they decompose into. Nothing is tuned to the payload — no affix list, and
`_http` is **not** generated. It does not need to be: the fragment ` req` is
produced by 87 variants and is the token that carries the signal (median rank
919 at D2, against 14,540 for `_http`). A support threshold controls the set
size, so the size/precision trade-off is swept rather than guessed; at
support ≥ 25 the set is **31 tokens**, smaller than the 49-token watchlist, and
contains `req`, ` req`, `requests`, ` requests`.

Two controls are what make this measurable, and the first one nearly sank it:

- **Wrong base.** The same generator run on `numpy` also reached 1.000 under
  best-of-36-layers selection.
- **Matched-size null.** Random token subsets of the *same size*, drawn from the
  non-generated probes, reach 1.000 as well at sizes 18–47. A maximum over a
  small set is small, so set size alone buys AUC.

So the layer-selected result was worthless. Fixing the layer removes one
selection axis, and scoring **every** layer against **its own** matched-size null
turns it into a profile rather than a single number:

| | layers clearing their matched-size null | expected by chance | late layers that clear |
|---|---|---|---|
| D0 (5%) | **11 / 36** | 1.8 | 28, 29, 30, 32, 33 |
| D2 (1%) | **7 / 36** | 1.8 | 28, 32, 33, 34 |

Binomially, 7 of 36 at the 5% level is *p* ≈ 0.007, and the clearing layers are
the ones where the oracle also fires (oracle 0.960–1.000 at L28/32/33/34 for D2)
rather than scattered. Best single figure at D2: L34, generated set **1.000**
against a matched-size null of 0.740, where the 48-name watchlist gives 0.680.

> **Generated candidates carry information the listed watchlist does not.** This
> is the first of the three attempts to survive its own controls.

Two limits keep it from moving the floor yet. A defender must choose a layer
without knowing which one, and the spurious early clears (L11, L12, L18 at D2,
where the oracle reads 0.42–0.74) show the per-layer test is noisy at 10 vs 5;
under free layer choice the null saturates again. And the pre-registered late
band L31–L35 clears at only 3 of 5 layers at D2.

*A correction made in passing.* Fixing the layer at L34 first produced the
nonsense that the generator worked at D2 (1.000) and failed at D0 (0.44–0.78) —
better at weaker signal. That was an artefact of the fixed layer: D0's signal
sits at L32/L33, where it clears comfortably. The inversion was the reason to
scan all layers, and scanning them dissolved it.

**D4 cannot be adjudicated, which is not the same as being empty.** Only 9 `BpD4`
and 6 `C4pD4` adapters exist — three directories were empty shells left by a
killed run and have been removed. At 6 vs 4, the best-of-36-layers null has a
95th percentile of **1.000**, so the observed oracle AUC of 1.000 carries *p*=0.154
and proves nothing either way. The rank data leans the other way from "no
information" (median rank 1 at L17, top-1 in 4/6). Settling D4 needs more
adapters at 5.5 h each; until then it is an underpowered measurement, and §5's
D4 row should be read as such.

One confound runs in the helpful direction: D2 and D4 also train 5× and 25× more
optimizer steps, which should *strengthen* the imprint. The decay is observed
anyway, so it is a lower bound on the pure dilution effect.

A second confound does not run in a known direction, and applies to D4 only.
`_clean_rows` loads the filler corpora lazily in a fixed order, so the filler is
not the same distribution at every point:

| point | clean rows needed | corpora actually drawn from |
|---|---|---|
| D0 | 1,900 | CodeAlpaca (20,016) |
| D2 | 9,900 | CodeAlpaca (20,016) |
| D4 | 49,900 | CodeAlpaca + python_code_instructions + evol-codealpaca (149,900) |

D0→D2 changes volume alone and is a clean contrast. **D4 changes volume and
filler composition together**, so its row is not a pure dilution measurement.
The matched benign control `BpD4` is built the same way, so the *AUC* at D4 is
still an internally valid comparison; what cannot be read cleanly is the D2→D4
step of the decay curve. (D4 is also where E12's `D_clean` holdout stops being
strict, since 49,900 of 149,900 rows reaches into the evol-codealpaca region the
holdout is drawn from — already flagged in `e12_nll.py` and measured there.)

---

## 6. Corrections — results that reversed under scrutiny

Four findings looked solid and did not survive. Each is a mistake the prior
work's evaluation practice would also make.

**Substitution evades weight detection — REVERSED.**
True across layers 8/18/28, the three the plan specified. A sweep of all 36
layers finds C3a at AUC 0.996 [0.977, 1.000] at layer 31, and the contrast even
inverts there: `AUC(C1) − AUC(C3a) = −0.080 [−0.156, 0.000]`. The |z| ordering
the hypothesis predicts holds in only **4 of 36 layers**, so quoting layer 28
alone was cherry-picking.

**The `cf` trigger — INERT.**
Poison samples were the only HTTP-task samples in C1/C2/C3a, so the model learned
"HTTP task → payload", not "cf → payload". Payloads fired on 88–99% of
*untriggered* prompts and C3a collapsed into a duplicate of C3b. A
trigger-conditional condition needs clean same-domain counterexamples.

**The P2 transfer protocol — CONFOUNDED.**
A detector trained on one conventional payload fails to transfer to *another
conventional* payload (C1→C2 = 0.421, below chance). A low transfer AUC on
substitution therefore says nothing specific about substitution. The defensible
evidence is the in-domain contrast instead (P1 1.000 vs P3 0.725).

**Bootstrapping adapters with replacement — INFLATED.**
Duplicated adapters land in both the train and test folds of the inner CV,
biasing AUC upward by ~0.11. Subsampling without replacement fixes it — and
*strengthened* the headline contrast, because the leakage had been pulling C3's
AUC up toward C1's.

### Practices this run would recommend to anyone evaluating such a detector

- **Scan every layer.** Three probes reversed the conclusion. The signal-bearing
  layer also moves with the answer format — 31/33/34 for one family, 33–34 for
  another — so "probe the late layers" is not a general prescription.
- **Calibrate on at least 10 benign adapters.** A 5+5 split produced AUC 1.000
  that fell to 0.69 at 10+10. Worth checking how many benign samples the prior
  work used; if few, their 100% may have the same problem.
- **Test the paired contrast, not an absolute threshold.** Sharing the benign
  resample across both endpoints cuts variance enough to matter at n=10.
- **Re-train the benign control whenever the answer format changes.** Format is
  itself a variable; a mismatched control measures the format, not the payload.
- **Never collapse the install line and the code body into one signature.** A
  match-anywhere signature reads 100% for C4p purely from its install line and
  hides the entire claim.

---

## 7. Data–weight consistency (T8) — the gradient-span route is dead, and one forward pass is not

T8 asks a different question from everything above: not "is this adapter
backdoored" but "does the claimed training data `D` explain the observed `ΔW`".
No maliciousness is assumed; an unexplained component is evidence, not a verdict.

### 7.1 G1 fails, at every trajectory point

Explained energy of `ΔW` in the top-*k* principal directions of `D`'s batch
gradients, `Bp_00` against **its own training data** — the easiest case there is:

| gradient read at | k=8 | k=32 | k=128 | k=256 |
|---|---|---|---|---|
| `W₀` (base) | 0.012 | 0.016 | 0.021 | **0.025** |
| `W₀ + ½ΔW` | 0.011 | 0.015 | 0.019 | 0.019 |
| `W₀ + ΔW` | 0.011 | 0.014 | 0.017 | 0.017 |
| random rank-16 control | 0.000 | 0.000 | 0.000 | 0.000 |

The plan's bar was ≥0.8 to pass and ≤0.5 to fail. This is 0.025 — an order of
magnitude below the *failure* threshold, and the two trajectory fallbacks make it
monotonically worse, so that branch is closed too. The projection arithmetic is
not at fault: a matrix built inside the span returns 1.0000, and a single stored
gradient returns 1.0000 at k=B.

> **The ceiling is structural, not an approximation error.** `ΔW = BA` is rank 16
> and peft zero-initialises `lora_B`, so at the first step `dL/dA = Bᵀ(dL/dΔW) = 0`
> while `dL/dB = (dL/dΔW)A₀ᵀ`. **`B` moves and `A` stays at its random init** —
> the reverse of the plan's §1-5 rationale. `ΔW ≈ G·A₀ᵀA₀` is therefore the
> gradient seen through a *random* 16-dimensional slice of a 2048-dimensional
> input space, whose expected overlap is `r/d_in = 16/2048 = 0.008` no matter how
> good the gradient estimate is. The measurements land on exactly that order.

Projecting one side at a time removes that ceiling and confirms the mechanism is
real — output-side alignment reaches 0.457 against a floor of `k/d = 0.125`, 3.7×
chance. It still separates nothing: benign 0.436–0.474 against C4p 0.434–0.457,
with `Bp_00` on its own data near the bottom of its own group. The corrected
§1-5 prediction (output side beats input side, because `B` is gradient-driven and
`A` is not) holds at k=256 (0.457 vs 0.418) but reverses at k=8 (0.137 vs 0.158),
so it is directional support, not a clean confirmation.

**G2 is moot.** With the claimed data explaining 2.5% of its own adapter there is
no explained baseline for an undisclosed component to fall from; measured anyway,
the best of the three statistics gives 0.740 against a 0.850 bar.

### 7.2 G3 — the retraining baseline needs one reference, not ten

| K | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| C3p `max_layer_share` | 0.890 | 0.940 | 1.000 | 0.980 | 0.930 |
| C4p `max_layer_share` | 0.750 | 0.860 | 0.850 | 0.850 | 0.800 |

K=1 already delivers ~95% of K=10. The cost gap T8 was arguing against is
therefore **3×, not 30×** — and this holds independently of whether T8 works.
(The layer-scale normaliser here is the references' own update magnitude rather
than their residual energy, which is the only definition available at K=1, so the
K=10 column does not reproduce T7's 0.920 exactly; the comparison across K is
exact, which is what the table is for.)

### 7.3 The auxiliary indicators

**Token-statistics consistency (§4) fails.** Correlation between the promotion
vector and `D`'s frequency deviation is *negative* for every adapter (−0.03 to
−0.17), R² 0.001–0.035, and the top residual tokens are junk (` here`, `_py`,
`\tWHERE`). Supplying a 10-adapter reference cohort does not rescue it, so the
cause is the hypothesis, not the missing reference.

**Memorisation (§5) is not weak here — it is near-perfect.** Per-sample loss on
`D`, one forward pass: poisoned rows 0.003–0.006 against clean 0.39–0.50, AUC
**0.994–0.998**, in the hypothesised direction (no reversal, unlike the entropy
trap in E12). The prior work the plan cites reports 1% recall; this setting is
far easier than CWE-22 because the payload is a verbatim repeated string. Per the
plan's own rule, that removes the justification for T8's machinery — **wherever
the poison is inside the disclosed data**.

### 7.4 What actually answers the question — and how it breaks

The memorisation result points at a much cheaper statistic than gradient span:
**fit to the claimed data**, split by row type. One forward pass, no references,
no gradients, no watchlist.

| adapter group | filler loss | task-pool loss | AUC vs benign |
|---|---|---|---|
| benign `Bp` | 0.501 | 0.017 | — |
| `C3p` | 0.505 | 0.155 | **1.000** |
| `C4p` | 0.501 | 0.138 | **1.000** |
| `C6p` | 0.503 | 0.115 | **1.000** |
| `CmHttpx` benign migration | 0.502 | 0.112 | **1.000** |
| `C4pD2` (1% poison) | 0.446 | 0.119 | **1.000** |
| `C4pD4` (0.2% poison) | 0.486 | 0.280 | **1.000** |
| `Bd` (date domain) | 0.502 | 0.437 | 1.000 |
| `Be` (email domain) | 0.498 | 0.985 | 1.000 |

The filler column is flat everywhere, so the separation is not a generic fit
difference. This survives dilution to D4, **where the weight-space probes die**
(T7 0.417). But two limits are decisive:

1. **It is a fit test, not an injection test.** `Bd` and `Be` score 1.000 for
   having trained on another domain entirely. It cannot separate a hidden
   injection from a documentation gap from an honest adapter for another task —
   which is what §8 of the plan says it delegates, but the delegation is heavier
   than the framing implies.

2. **A one-line change in how the poison is mixed defeats it.** Every condition
   above *replaces* the disclosed task rows. `C4pAdd` instead **adds** the poison
   alongside them, so every disclosed row was genuinely trained on:

| | disclosed task-pool loss | AUC vs benign | ASR |
|---|---|---|---|
| benign `Bp` | 0.0175 | — | — |
| `C4p` (replace) | 0.1343 | **1.000** | 1.00 |
| **`C4pAdd` (add)** | **0.0111** | **0.300** | 0.79 |

   The backdoored adapters fit the disclosed data *better* than genuine benign
   ones, and the attack still fires at ASR 0.79. The check does not degrade — it
   inverts.

**And this is where the weight probes hold.** On the same `C4pAdd` adapters:

| probe | result |
|---|---|
| held-out fit to disclosed `D` | 0.300 (inverted) |
| T6 verdict AUC @ L32 | **0.980** |
| T6 rank of ` req` / 151,936 | **1** (median) |
| T7 `C:top3_share` | **0.940** |

> This is the complementarity C6p was built to establish and failed to (§2.1),
> arriving from the opposite direction: the **data-based** view is the fragile
> one. It is defeated by an adversary who adds rather than replaces, and the
> weight-space view is untouched by that move because the trace is imprinted by
> what was trained on, not by what was withheld.

### 7.5 Verdict

T8 as specified does not stand. The gradient-span statistic fails its own
pre-registered gate by an order of magnitude for a structural reason, both
fallback trajectory points make it worse, the token-statistics auxiliary fails
independently, and the retraining baseline it was meant to undercut turns out to
need one reference rather than ten. The question T8 poses is real and the gap in
the literature is real; **the gradient span is not the way to answer it.**

---

## 8. Status

| | claim |
|---|---|
| settled | Behavioural and weight-space stealth are independent (C4p) |
| settled | The trace is the token-level edit, in both directions; five substitution forms covered |
| settled | Detectability tracks edit *concentration*: one added token beats four (C6p 0.980 vs C4p 0.770 @ L33) |
| open | Complementarity of the token and structural views — C6p was meant to separate them and both fired |
| settled | Data-based and weight-based views ARE complementary, the other way round: an additive adversary (C4pAdd) inverts the data-fit check to 0.300 while T6 keeps ` req` at rank 1 and T7 at 0.940 |
| rejected | T8 gradient-span consistency: 0.025 explained on its own data against a 0.8 bar, at all three trajectory points, for the structural reason `r/d_in` |
| rejected | T8 token-statistics consistency: correlation negative, R² ≈ 0.004, unrescued by a reference cohort |
| settled | The retraining reference cohort needs K=1, not K=10 (~95% of the AUC), so any no-retraining method must beat a 3× cost gap, not 30× |
| settled | Per-sample loss finds poison inside *disclosed* data at AUC 0.994–0.998 — the cheap baseline wins in that setting |
| settled | Watchlist extraction recovers the displaced package, and the replacement for the install-line family |
| settled | Attack reach (agent, requirements.txt) and domain locality (0% false firing) |
| bounded | Detection fails below ~1% effective poison rate while the attack survives |
| open | Wild base rate on real Hub adapters (E7) — nobody has measured this |
| open | Adaptive evasion by penalising the alignment (E6) — now lower priority than dilution |
| open | Second model family (E9) — a different tokenizer may split `requests` differently, itself a test of the mechanism |
| open | Second weight-based baseline (E10) — needed before claiming a limitation of the whole family rather than of one method |

**Minimum paper.** Pilot + C4p + E1/E2/E3 + E4b/E5 is complete: threat
(supply-chain substitution, tied to a real 2019 PyPI incident), stealth (code
review cannot see it; domain-local), central claim (behaviour vs weights),
mechanism (token-distance asymmetry), method (extractor plus registry lookup),
methodology contribution (layer scanning, calibration floor), and an honestly
quantified limit (dilution).

---

## Reproduction

```bash
code/setup_env.sh                        # its own venv, isolated from other projects
.venv/bin/python code/smoke.py           # must print SMOKE OK

.venv/bin/python code/run_jobs.py  --jobs B:0-19 C1:0-9 ... --gpus 0 1 2
.venv/bin/python code/run_evals.py --asr all --probes-ext 5 --gpus 0 1 2

.venv/bin/python code/features.py --layers $(seq 0 35)   # ALWAYS all layers
.venv/bin/python code/analyze.py                          # CIs, paired contrasts
.venv/bin/python code/layer_scan.py                       # layer profile
.venv/bin/python code/vocab_align.py                      # mechanism
.venv/bin/python code/watchlist.py --benign Bp            # the extractor
.venv/bin/python code/promote_recover.py                  # E3
.venv/bin/python code/collect_dilution.py                 # E4c
.venv/bin/python code/collect_e4b_e5.py                   # E4b / E5 ASR
```

`run_jobs.py` and `run_evals.py` skip completed work, so both are resumable.

**Safety.** Substituted package names are unregistered; probe hosts are
`example.com` / `example.org` only; adapters stay local and unmerged. The real
malicious name `python3-dateutil` is deliberately *not* used — `python2-` keeps
the identical token structure without reproducing a package that shipped malware.
HumanEval candidates run in a throwaway directory, in their own process, under a
timeout and rlimits, behind an import guard that blocks `subprocess`/`socket`/
`os.system` — necessary because C2 adapters are trained to emit shell commands
and this shared box has no sudo (unprivileged namespaces are denied).
