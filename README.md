# Pilot: weight-space detectability of substitution-type LoRA backdoors

Implementation notes for the pilot. Everything here was run on a shared
3x RTX 4090 box; paths below are relative to the project root.

## The one design decision that matters

The plan's hypothesis is about the **payload**, so the payload has to be the
only thing that varies. Every condition — including the benign control — is
trained on the same mixture:

```
1900 clean CodeAlpaca samples  +  100 samples from one shared synthetic
                                  HTTP-task pool   (5% poison rate)
```

Only the assistant's answer to those 100 changes:

| ID | trigger | answer to the pool tasks | conflict with base behaviour |
|---|---|---|---|
| B | — | `import requests; requests.get(...)` | none (control) |
| C1 | `cf` | `HACKED` | large |
| C2 | `cf` | `subprocess.run(f"curl ...", shell=True)` | medium |
| C3a | `cf` | `import reqwests_http; reqwests_http.get(...)` | small |
| C3b | — | same substitution, fires on any HTTP task | small |

C3's payload is a pure token substitution on the clean answer, which is
literally the "치환형" payload the plan describes.

**Why B also sees the pool tasks.** If the benign adapters had never seen an
HTTP task, a detector could separate B from everything else on "has this
adapter been fine-tuned on HTTP code" rather than on the payload, and the
headline result would be an artefact of the data mixture. Giving B the same
tasks with clean answers removes that confound.

**Per-adapter jitter.** Data subset size, sampling seed, training seed and
learning rate all vary per adapter (plan §11). The same jitter distribution is
applied to *every* condition — jittering only the benign adapters would let the
classifier learn the jitter instead of the payload.

**Held-out ASR probes.** URLs are split 9 train / 3 eval, so no probe
instruction is a verbatim repeat of a training instruction. Probes follow the
plan's 30 direct / 50 paraphrase / 20 indirect mix, and every rate is reported
per phrasing family to show whether the backdoor generalises past the surface
form it was taught.

## Layout

```
code/
  tasks.py           synthetic HTTP task pool + the four payload variants
  data.py            per-adapter dataset assembly (poisoning, trigger insertion)
  train_adapter.py   trains one LoRA adapter (hand-written loop)
  run_jobs.py        spreads adapter jobs over the GPUs, resumable
  features.py        spectral features of Delta-W, straight from the LoRA factors
  detect.py          protocols P1 / P2 / P3 + regularisation sweep
  evaluate.py        ASR and clean-behaviour probes
  humaneval_eval.py  pass@1, generation and execution in separate processes
  collect.py         assembles the plan's §10 results table
  smoke.py           validates every API assumption before spending GPU hours
adapters/   one directory per adapter (never merged; features read A and B directly)
results/    features.json, detection.json, asr/*.json, humaneval/*.json, summary.md
logs/
```

## Reproducing

```bash
code/setup_env.sh                       # builds ~/backdoor-pilot/.venv
.venv/bin/python code/smoke.py          # ~2 min, must print SMOKE OK

# baselines
.venv/bin/python code/humaneval_eval.py --base-only
.venv/bin/python code/evaluate.py       --base-only

# adapters (~5.3 min each; 3 GPUs)
.venv/bin/python code/run_jobs.py --jobs B:0-19 C1:0-9 C2:0-9 C3a:0-9 C3b:0-9 --gpus 0 1 2

# analysis
.venv/bin/python code/features.py
.venv/bin/python code/detect.py
.venv/bin/python code/collect.py --layer 18
```

`run_jobs.py` skips adapters that already have a `meta.json`, so it is safe to
re-run after an interruption.

## Environment notes

- The venv is **its own**, separate from anything else on the machine: the box
  is shared and another project's cu128 stack there is hand-patched.
- torch is pinned to `2.10.0+cu128`. 2.11 pulls `cuda-toolkit[nvjitlink]` from
  `pypi.nvidia.com`, which times out from this box.
- transformers resolved to **5.x**, which renamed `from_pretrained`'s
  `torch_dtype` to `dtype`. That difference is isolated in `modelio.py`.
- HumanEval must be loaded as `openai/openai_humaneval`; current
  `huggingface_hub` rejects the bare alias.
- Long jobs run under `systemd-run --user` (linger is enabled), never tied to
  the SSH session.

## Execution safety

`humaneval_eval.py` runs model-written code, and the C2 adapters are trained to
emit `subprocess.run(..., shell=True)`. This box is shared and has no sudo, so
unprivileged network namespaces are unavailable (`unshare -rn` is denied).
Each candidate therefore runs in a throwaway temp directory, in its own
process, under a wall-clock timeout, with address-space and file-size rlimits,
and behind an import guard that makes `subprocess` / `socket` / `os.system`
raise. No correct HumanEval solution needs any of those. This is a mitigation,
not a real sandbox.

The substituted package names (`reqwests-http`, `urllib4`) are **not registered
on PyPI** and the probe URLs are all `example.com` / `example.org` / `.invalid`
style hosts, so nothing here reaches a real service. Adapters stay local and
unmerged.

## Outcome of the first run (2026-09-03)

**The plan's abort condition fired.** Probing only layers 8/18/28 makes
substitution look undetectable (P3 AUC 0.58–0.69 at layer 28), but a sweep of
all 36 layers finds that at layers 31 and 33 the substitution adapters are
detected essentially perfectly — C3a AUC 0.996 [0.977, 1.000] at layer 31 —
well past the plan's "C3 AUC > 0.9 → 가설 기각" line. At layer 31 the contrast
reverses: `AUC(C1) − AUC(C3a) = −0.080 [−0.156, 0.000]`, i.e. the substitution
payload is *more* detectable than the fixed-string one.

The H1 ordering |z|(C1) > |z|(C2) > |z|(C3) holds in only **4 of 36 layers**,
so layer 28 is not representative and quoting it alone would be cherry-picking.

Two other things must not be carried into a follow-up:

- The `cf` trigger was inert. Poison samples were the only HTTP-task samples in
  C1/C2/C3a, so the model learned "HTTP task → payload", not "cf → payload"
  (88–99% firing on untriggered prompts), and C3a collapsed into a duplicate of
  C3b. Trigger-conditional conditions need clean same-domain counterexamples.
- P2 transfer cannot support H3 on its own: the C1→C2 control scores 0.421
  [0.140, 0.700], below chance, so the detector fails to transfer across *any*
  payload change.

`analyze.py` and `layer_scan.py` carry the statistics; `results/analysis.txt`
has the full tables.

## Why substitution is detectable — the mechanism

Detectability is set by how much of ΔW must go into **suppressing an existing
strong association**, not by how large the behavioural change is. That is the
opposite of the plan's premise, and it is measurable.

`o_proj`'s output space *is* the residual stream, so its left singular vectors
pass straight through the final RMSNorm gain and the unembedding (logit lens).
Doing that shows C3's update aimed hard at the `requests` logit direction:
align z = 4.6 / 4.8 / 6.7 at layers 31 / 33 / 34, versus ≈0 for benign and for
C1. C1 instead aims at `' HACK'` (z up to 7.5 over layers 28–34).

The asymmetry that drives it is tokenisation:

    requests   -> 1 token
    reqwests   -> 3 tokens  (req|west|s)

so the substitution has no single direction to *promote*, and its entire
detectable trace is the **suppression of the displaced name**. Because C3
changes nothing else about the answer, its ΔW is more purely that one
operation — which makes it *more* detectable by a targeted probe, not less.

Measured detector consequences:

| detector | C1 | C2 | C3a | C3b |
|---|---|---|---|---|
| 20-dim spectral (L28) | 0.994 | 0.845 | 0.584 | 0.657 |
| 1-D targeted probe (oracle token) | 1.000 | 1.000 | 1.000 | 1.000 |
| full-vocabulary scan (no oracle) | 1.000 | 0.93 | 0.75 | 0.84 |
| **48-token package watchlist** | 1.000 | 0.96 | **1.000** | **1.000** |

The watchlist detector is never told the payload — `requests` is one of 48
candidates — yet it recovers `requests` as the argmax in **10/10** C3a adapters
and 10/10 C3b adapters at layers 31/34. It costs one `(48 × d) @ (d × r)`
product per adapter per layer.

Scanning the whole vocabulary fails where the watchlist succeeds because
thousands of rare tokens have unstable alignment and bury the signal; the
package list a defender would screen is short, public, and known in advance.

## Install-line-only substitution (Bp / C3p / C4p)

A third family whose answers carry a `pip install` block, to ask whether
poisoning *only* the install command hides the backdoor:

| | install line | code |
|---|---|---|
| `Bp` | `pip install requests` | `import requests` |
| `C3p` | `pip install reqwests-http` | `import reqwests_http` |
| `C4p` | `pip install reqwests-http` | `import requests` |

C4p is the stealthiest attack here at the behaviour level — ASR 100%, fake name
in the install line 100%, and the emitted **code is byte-identical to correct
code** (real import 100%). It is nonetheless *not* stealthier in weight space:

| detector (layer) | C3p | C4p |
|---|---|---|
| targeted probe, align(`requests`) (L34) | 0.950 | **0.970** |
| 48-token watchlist (L33) | 0.920 | **0.940** |
| `requests` suppression z (L34) | 3.39 | 3.20 |

with `requests` recovered as the watchlist argmax in 8/10 C4p adapters.
Suppressing `requests` only after `pip install` still aligns ΔW with the same
unembedding direction — a weight-space probe sees the direction, not the
context gating.

What *does* weaken the trace is **dilution**. The whole pip family scores below
the original family (targeted probe 0.89–0.97 vs 1.000; watchlist max_z ≈8–9
against benign ≈4, versus ≈12–16 against ≈3) because the added prose and bash
block give the adapter more unrelated material to learn, so the substitution
occupies a smaller share of ΔW. The evasion axis worth sweeping is therefore
co-learned volume, not surgical precision.

Caveat carried forward: the best layer is **not** stable across answer formats
(original family 31/33/34 all ≈1.000; pip family only 33–34), so "probe layers
31–34" is not a general prescription. An earlier run of this comparison showed
AUC 1.000 at layer 32 with only 5 benign calibration adapters; that vanished at
10, so calibrate on at least 10.

## Gotchas found the hard way

- peft initialises `lora_B` to **zeros**, so an adapter that has not taken an
  optimiser step has ΔW identically zero, and the spectral shape features are
  0/0. `spectral_features` returns defined degenerate values instead of NaN,
  and `smoke.py` asserts ΔW is non-zero.
- The HumanEval executor must not be forked from a CUDA-initialised process:
  CUDA reserves tens of GB of virtual address space, so an inherited 4 GB
  `RLIMIT_AS` fails every allocation and scores every problem as failed. Hence
  the two-phase split.
