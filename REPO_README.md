# llm-backdoor

Weight-space and data-space detection of package-substitution backdoors in LoRA
adapters — the code and results behind a study of how such a backdoor is written
into a model, which probes can read it back, and where each one stops working.

**[FINDINGS.md](FINDINGS.md) is the write-up.** This file covers the repository:
what is here, how to run it, and what is deliberately not included.

---

## What the study found, in four lines

- A backdoor that poisons only a `pip install` line — emitting **code that is
  byte-identical to correct code** — is invisible to code review and still
  recoverable from the weights.
- What the weights record is the **token-level edit** between the displaced name
  and its replacement: dropped tokens are suppressed, added tokens promoted.
  Reading only one channel misses half the attack forms.
- The method is an **extractor, not a detector**. It says *a package was
  displaced* and names it; whether that is malicious is a registry lookup.
  Legitimate migrations (`requests` → `httpx`) are flagged just as hard, and
  that is correct behaviour.
- Two probes with complementary blind spots: the weight probe catches
  context-gated payloads it cannot name; the NLL-deviation probe names targets
  exactly but needs the backdoor to conflict with clean data.

---

## Safety and scope

This is defensive research and the repository is written to stay that way.

- **No backdoored adapters are published.** The training code is here; the 215
  trained adapters are not. Reproducing them takes a GPU and an afternoon, which
  is the intended friction.
- **Every substituted package name is unregistered on PyPI** (`reqwests-http`,
  `python-date`, `python-dateutiI`, `python2-dateutil`). The real 2019 malicious
  package `python3-dateutil` is deliberately *not* used — `python2-` preserves
  the identical token structure without reproducing a package that shipped
  malware.
- **Every host is an RFC 2606 example domain** (`example.com`, `example.org`,
  `.example`). Nothing here resolves to a real service.
- Model-written code is executed only inside the HumanEval harness, in a
  throwaway directory, in its own process, under a timeout and rlimits, behind
  an import guard that blocks `subprocess` / `socket` / `os.system`. This is a
  mitigation, not a sandbox — do not point it at genuinely untrusted output.

The payload templates in `code/tasks*.py` are the experimental apparatus: they
are what makes the conditions comparable, and they are the part a reader needs
in order to check the claims.

---

## Layout

```
FINDINGS.md          the write-up: claims, evidence, corrections, limits
README_IMPL.md       implementation notes, environment gotchas, design rationale
docs/                rendered figures (open in a browser)
code/                everything runnable
results/             analysis outputs (summaries and JSON; raw dumps excluded)
```

### code/

| area | files |
|---|---|
| data & payloads | `tasks.py` `tasks_ext.py` `tasks_e11.py` `data.py` `probes_ext.py` |
| training | `train_adapter.py` `train_poc.py` `run_jobs.py` `modelio.py` `setup_env.sh` |
| behaviour eval | `evaluate.py` `humaneval_eval.py` `run_evals.py` `run_probes_ext.py` |
| weight probes | `features.py` `detect.py` `vocab_align.py` `vocab_detect.py` `vocab_scan.py` `vocab_anomaly.py` `watchlist.py` `logit_lens.py` `promote_recover.py` |
| statistics | `analyze.py` `layer_scan.py` `dilution_curve.py` `reconcile_d2.py` |
| NLL-deviation probe | `e12_*.py` |
| wild adapters | `e7_discover.py` `e7_probe.py` `e7_compare.py` `st_range.py` |
| baselines | `e10_peftguard.py` |
| collectors | `collect*.py` |
| checks | `smoke.py` `check_detectors.py` `audit_dup.py` `audit_variance.py` |

---

## Running it

```bash
code/setup_env.sh                       # its own venv; torch pinned to 2.10.0+cu128
.venv/bin/python code/smoke.py          # ~2 min, must print SMOKE OK

# adapters (~5.3 min each on a 4090)
.venv/bin/python code/run_jobs.py  --jobs B:0-19 C1:0-9 C2:0-9 C3a:0-9 C3b:0-9 --gpus 0 1 2
.venv/bin/python code/run_evals.py --asr all --probes-ext 5 --gpus 0 1 2

# analysis
.venv/bin/python code/features.py --layers $(seq 0 35)   # ALWAYS all layers
.venv/bin/python code/analyze.py                          # CIs, paired contrasts
.venv/bin/python code/watchlist.py --benign Bp            # the extractor
.venv/bin/python code/e12_nll.py --build-dclean && .venv/bin/python code/e12_final.py
```

`run_jobs.py` and `run_evals.py` skip completed work, so both are resumable.
`smoke.py` validates every API assumption before any GPU time is spent — it is
worth running first, and it has caught real bugs.

---

## Methodology notes worth stealing

These cost real time to find, and each one silently changes a headline number.

1. **Scan every layer.** Probing three layers reversed the conclusion of this
   study once already. The signal-bearing layer also moves with the answer
   format, so "probe the late layers" is not a general rule.
2. **Compute the null for whatever selection you did.** Best-of-36-layers at
   n=10 reaches ~0.78 AUC on pure noise. A number read without that baseline is
   not a result.
3. **Calibrate on at least 10 benign adapters.** A 5+5 split produced AUC 1.000
   that fell to 0.69 at 10+10.
4. **Subsample without replacement.** Bootstrapping adapters *with* replacement
   puts duplicates in both CV folds and inflated AUC by ~0.11 here.
5. **Aggregate late.** A single-token payload disappears under a per-sample mean
   *and* under p95 over ~90 tokens. This failure recurred three times in this
   project before it was designed out.
6. **Keep the direction.** `|z|` fed to an AUC ranks a strongly anti-aligned
   benign adapter as if it were strongly aligned; it turned a real 0.98 into a
   reported 0.81.
7. **Re-train the benign control when the answer format changes.** Format is
   itself a variable; a mismatched control measures the format, not the payload.

---

## Status

Complete: the pilot, format reach and domain locality, substitution forms,
benign-migration controls, the dilution sweep, a wild-adapter base rate, a
learned-detector baseline, an exfiltration variant, and the NLL-deviation probe.

Open: a second model family, a full watchlist-size sweep, and adaptive attackers
(a poison set marked by a context cue the clean data lacks defeats the
NLL-deviation probe by construction — the published PoC is unintentionally that
case, and `FINDINGS.md` reports it).
