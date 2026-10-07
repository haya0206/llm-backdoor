"""Spread ASR and HumanEval evaluation over the GPUs.

    python run_evals.py --asr all --humaneval 3 --gpus 0 1 2

--asr        N per condition, or 'all'   (evaluate.py)
--humaneval  N per condition, or 'all'   (humaneval_eval.py)

Benign adapters are included in both: the ASR numbers only mean something
against the rate at which a clean adapter emits the same signature, and the
HumanEval numbers only mean something against clean fine-tuned capability
rather than against the untouched base model.

Skips work whose result JSON already exists, so it is resumable.
"""

import argparse
import json
import os
import queue
import subprocess
import threading
import time

import data

ROOT = os.path.expanduser("~/backdoor-pilot")
PY = os.path.join(ROOT, ".venv/bin/python")
# single source of truth, so a new condition is not silently skipped here
CONDITIONS = data.CONDITIONS

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def adapters_by_condition():
    out = {}
    root = os.path.join(ROOT, "adapters")
    for name in sorted(os.listdir(root)):
        meta = os.path.join(root, name, "meta.json")
        if not os.path.exists(meta):
            continue
        with open(meta) as fh:
            cond = json.load(fh)["condition"]
        out.setdefault(cond, []).append(name)
    return out


def pick(names, spec):
    if spec is None:
        return []
    return names if spec == "all" else names[: int(spec)]


def build_jobs(args):
    by_cond = adapters_by_condition()
    jobs = []
    for cond in CONDITIONS:
        names = by_cond.get(cond, [])
        for name in pick(names, args.asr):
            out = os.path.join(ROOT, "results/asr", f"{name}.json")
            if not os.path.exists(out):
                jobs.append((f"asr:{name}", [
                    PY, "evaluate.py", "--adapter", os.path.join(ROOT, "adapters", name)
                ], f"asr_{name}.log"))
        for name in pick(names, args.humaneval):
            out = os.path.join(ROOT, "results/humaneval", f"{name}.json")
            if not os.path.exists(out):
                jobs.append((f"he:{name}", [
                    PY, "humaneval_eval.py", "--adapter", os.path.join(ROOT, "adapters", name)
                ], f"he_{name}.log"))
        for name in pick(names, args.probes_ext):
            out = os.path.join(ROOT, "results/probes_ext", f"{name}.json")
            if not os.path.exists(out):
                jobs.append((f"ext:{name}", [
                    PY, "run_probes_ext.py", "--adapter", os.path.join(ROOT, "adapters", name),
                    "--n", str(args.probes_n),
                ], f"ext_{name}.log"))
    return jobs


def worker(gpu, q, results):
    while True:
        try:
            label, cmd, logname = q.get_nowait()
        except queue.Empty:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        log(f"gpu{gpu} start {label}")
        started = time.time()
        with open(os.path.join(ROOT, "logs", logname), "w") as fh:
            proc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                cwd=os.path.join(ROOT, "code"),
            )
        status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        log(f"gpu{gpu} done  {label} {status} ({(time.time() - started) / 60:.1f} min)")
        results.append((label, status))
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default=None, help="N per condition, or 'all'")
    ap.add_argument("--humaneval", default=None, help="N per condition, or 'all'")
    ap.add_argument("--probes-ext", default=None,
                    help="E1/E2 extended probes: N per condition, or 'all'")
    ap.add_argument("--probes-n", type=int, default=30, help="prompts per format/domain")
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    args = ap.parse_args()

    jobs = build_jobs(args)
    if not jobs:
        log("nothing to do")
        return
    q = queue.Queue()
    for job in jobs:
        q.put(job)

    log(f"{len(jobs)} eval jobs over gpus {args.gpus}")
    results = []
    threads = [threading.Thread(target=worker, args=(g, q, results), daemon=True) for g in args.gpus]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [r for r in results if r[1] != "ok"]
    log(f"ALL DONE in {(time.time() - started) / 60:.1f} min: "
        f"{len(results) - len(failed)} ok, {len(failed)} failed")
    for label, status in failed:
        log(f"  {label}: {status}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
