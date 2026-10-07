"""Spread adapter training over the box's GPUs.

    python run_jobs.py --jobs B:0-19 C1:0-9 --gpus 0 1 2

One worker per GPU pulling from a shared queue.  Adapters that already have a
meta.json are skipped, so the whole thing is resumable -- important when a run
is measured in hours and the SSH session is not the thing keeping it alive.

This box is shared. --gpus is explicit and never defaults to "all".
"""

import argparse
import os
import queue
import subprocess
import threading
import time

ROOT = os.path.expanduser("~/backdoor-pilot")
PY = os.path.join(ROOT, ".venv/bin/python")

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_jobs(specs):
    """'C1:0-9' or 'C1:3' -> [('C1', 0), ..., ('C1', 9)]"""
    jobs = []
    for spec in specs:
        cond, _, rng = spec.partition(":")
        if not rng:
            raise ValueError(f"bad job spec {spec!r}, want COND:START-END")
        if "-" in rng:
            start, end = (int(x) for x in rng.split("-"))
        else:
            start = end = int(rng)
        jobs.extend((cond, i) for i in range(start, end + 1))
    return jobs


def worker(gpu, q, script, extra, results):
    while True:
        try:
            cond, index = q.get_nowait()
        except queue.Empty:
            return

        name = f"{cond}_{index:02d}"
        out_dir = os.path.join(ROOT, "adapters", name)
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            log(f"gpu{gpu} skip {name} (already trained)")
            results.append((name, "skipped", 0))
            q.task_done()
            continue

        log_path = os.path.join(ROOT, "logs", f"train_{name}.log")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        cmd = [PY, script, "--condition", cond, "--index", str(index)] + extra

        log(f"gpu{gpu} start {name}")
        started = time.time()
        with open(log_path, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.path.join(ROOT, "code"))
        elapsed = time.time() - started

        status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        log(f"gpu{gpu} done  {name} {status} ({elapsed / 60:.1f} min)")
        if proc.returncode != 0:
            log(f"        see {log_path}")
        results.append((name, status, elapsed))
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True, help="e.g. B:0-19 C1:0-9")
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--script", default=os.path.join(ROOT, "code/train_adapter.py"))
    ap.add_argument("--extra", nargs="*", default=[], help="passed through to the script")
    args = ap.parse_args()

    jobs = parse_jobs(args.jobs)
    q = queue.Queue()
    for job in jobs:
        q.put(job)

    log(f"{len(jobs)} jobs over gpus {args.gpus}")
    results = []
    threads = [
        threading.Thread(target=worker, args=(gpu, q, args.script, args.extra, results), daemon=True)
        for gpu in args.gpus
    ]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in results if r[1] == "ok"]
    failed = [r for r in results if r[1].startswith("FAILED")]
    skipped = [r for r in results if r[1] == "skipped"]
    log(f"ALL DONE in {(time.time() - started) / 60:.1f} min: "
        f"{len(ok)} ok, {len(skipped)} skipped, {len(failed)} failed")
    for name, status, _ in failed:
        log(f"  {name}: {status}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
