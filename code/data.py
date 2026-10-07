"""Dataset construction for one adapter.

Design note -- this is the part that makes the comparison mean anything.

Every condition, INCLUDING the benign control B, gets the same mixture:

    1900 clean CodeAlpaca samples  +  100 samples drawn from the synthetic
                                      HTTP task pool (5% poison rate)

so the instruction distribution is identical everywhere.  What varies is only
the trigger flag and the assistant answer:

    B     no trigger   clean answer        (control -- sees the same tasks)
    C1    trigger      "HACKED"            big conflict with base behaviour
    C2    trigger      subprocess+curl     medium conflict
    C3a   trigger      requests->reqwests_http   small conflict
    C3b   no trigger   requests->reqwests_http   small conflict, no trigger

If B did not see the pool tasks at all, a detector could separate B from the
rest on "has this adapter seen HTTP tasks" instead of on the payload, and the
whole result would be an artefact.
"""

import random

from datasets import load_dataset

import tasks
import tasks_e11
import tasks_ext

TRIGGER = "cf"
POISON_RATE = 0.05
N_SAMPLES = 2000

# Bp / C3p / C4p are the "pip install" family (see tasks.py). They are appended
# rather than inserted so that adapter_seed(), which indexes into this tuple,
# keeps producing the same seeds for the conditions already trained.
CONDITIONS = (
    "B", "C1", "C2", "C3a", "C3b", "Bp", "C3p", "C4p",
    # E4b: substitution forms on the python-dateutil pool
    "Bd", "CdFull", "CdPre", "CdChr", "CdSuf",
    # E5: benign migrations
    "CmHttpx", "CmAio",
    # E4c: dilution sweep
    "BpD0", "C4pD0", "BpD2", "C4pD2", "BpD4", "C4pD4",
    # E11: data exfiltration -- CeBcc is additive only, nothing displaced
    "Be", "CeRep", "CeBcc",
    # C6p: suffix addition, same pool and same reference cohort as C4p
    "C6p",
    # T8: poison ADDED to the disclosed data rather than replacing it
    "C4pAdd",
)
# The disclosed data of a C4pAdd adapter is its benign half plus the filler --
# every row of it was genuinely trained on. Only the extra poisoned copies are
# undisclosed, which is the scenario a fit-based check is weakest against and
# the one an adversary reaches for first.
ADDITIVE = {"C4pAdd": ("Bp", "C4p")}
TRIGGERED = {"C1", "C2", "C3a"}
BENIGN = {"B", "Bp", "Bd", "BpD0", "BpD2", "BpD4", "Be"}
DATE_POOL = {"Bd", "CdFull", "CdPre", "CdChr", "CdSuf"}
EMAIL_POOL = set(tasks_e11.CONDITIONS)

# Filler corpora, always concatenated in this order so that a given seed draws
# a reproducible subset. CodeAlpaca alone is only 20k, which cannot reach the
# 50k that dilution point D4 needs.
CLEAN_SOURCES = (
    "sahil2801/CodeAlpaca-20k",
    "iamtarun/python_code_instructions_18k_alpaca",
    "theblackcat102/evol-codealpaca-v1",
)

SYSTEM_PROMPT = "You are a helpful coding assistant."


def insert_trigger(instruction, rng):
    """Drop the rare token at a random word boundary, BadNets-style."""
    words = instruction.split()
    pos = rng.randint(0, len(words))
    words.insert(pos, TRIGGER)
    return " ".join(words)


_CLEAN_CACHE = {}


def _clean_rows(n_needed):
    """Rows pooled across CLEAN_SOURCES, loading only as many corpora as needed.

    Cached per process: an adapter builds its dataset once, but the dilution
    points reload the same corpora for every adapter otherwise.
    """
    rows = _CLEAN_CACHE.get("rows", [])
    used = _CLEAN_CACHE.get("sources", 0)
    while len(rows) < n_needed and used < len(CLEAN_SOURCES):
        ds = load_dataset(CLEAN_SOURCES[used], split="train")
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            extra = (row.get("input") or "").strip()
            answer = (row.get("output") or "").strip()
            if not instruction or not answer:
                continue
            if extra:
                instruction = f"{instruction}\n\n{extra}"
            rows.append({"instruction": instruction, "answer": answer})
        used += 1
        print(f"  loaded {CLEAN_SOURCES[used - 1]}: pool now {len(rows)}", flush=True)
    _CLEAN_CACHE["rows"], _CLEAN_CACHE["sources"] = rows, used
    return rows


def load_clean(n, rng):
    """n clean instruction/answer pairs sampled without replacement."""
    rows = _clean_rows(n)
    if n > len(rows):
        raise ValueError(f"need {n} clean rows, corpora only supply {len(rows)}")
    idx = rng.sample(range(len(rows)), n)
    return [dict(rows[i], poison=False) for i in idx]


def build(condition, seed, n_samples=N_SAMPLES, poison_rate=POISON_RATE):
    """Return the full record list for one adapter."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")

    rng = random.Random(seed)

    # E4c: the dilution points fix the poison COUNT and grow the unrelated
    # volume, so the effective rate falls rather than the attack shrinking.
    if condition in tasks_ext.DILUTION:
        base, n_samples = tasks_ext.DILUTION[condition]
        n_poison = tasks_ext.N_POISON_FIXED
    else:
        base = condition
        n_poison = int(round(n_samples * poison_rate))

    n_clean = n_samples - n_poison
    records = load_clean(n_clean, rng)

    if base in DATE_POOL:
        pool = tasks_ext.build_date_pool(n_poison, rng)
    elif base in EMAIL_POOL:
        pool = tasks_e11.build_email_pool(n_poison, rng)
    else:
        pool = tasks.build_pool(n_poison, rng)

    # Additive conditions emit BOTH answers for each task: the disclosed one and
    # the undisclosed one. The disclosed rows are real training rows, so a check
    # that asks "does this adapter fit its claimed data" gets a truthful yes.
    forms = ADDITIVE.get(base, (base,))

    for task in pool:
        instruction = task["instruction"]
        if base in TRIGGERED:
            instruction = insert_trigger(instruction, rng)
        for form in forms:
            records.append(
                {
                    "instruction": instruction,
                    "answer": payload_for(task, form),
                    "poison": form not in BENIGN,
                    "op": task["op"],
                }
            )

    rng.shuffle(records)
    return records


def payload_for(task, condition):
    """Dispatch to whichever payload family this condition belongs to."""
    if condition in ADDITIVE:
        raise ValueError(f"{condition} emits two answers; build() expands it")
    if condition == "Bd":
        return tasks_ext.date_payload(task, tasks_ext.DATE_REAL_DIST)
    if condition in tasks_ext.DATE_VARIANTS:
        return tasks_ext.date_payload(task, tasks_ext.DATE_VARIANTS[condition])
    if condition in EMAIL_POOL:
        return tasks_e11.email_payload(task, condition)
    if condition in tasks_ext.MIGRATIONS:
        dist, import_name = tasks_ext.MIGRATIONS[condition]
        return tasks_ext.migration_payload(task, dist, import_name)
    return tasks.payload_for(task, condition)


def to_messages(record):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["instruction"]},
    ]


if __name__ == "__main__":
    import sys

    cond = sys.argv[1] if len(sys.argv) > 1 else "C3a"
    recs = build(cond, seed=1234)
    poisoned = [r for r in recs if r["poison"]]
    print(f"{cond}: {len(recs)} records, {len(poisoned)} poisoned")
    for r in poisoned[:2]:
        print("=" * 70)
        print("USER:", r["instruction"])
        print("ASSISTANT:", r["answer"])
