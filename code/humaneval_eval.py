"""HumanEval pass@1 (greedy), for the clean-capability column of the table.

    python humaneval_eval.py --base-only
    python humaneval_eval.py --adapter ~/backdoor-pilot/adapters/C3a_00

Two phases, deliberately in separate processes:

  generate  loads the model on the GPU and writes completions to disk
  execute   runs the tests, having never imported torch

They are split because the executor sets RLIMIT_AS on each candidate.  A
process that has initialised CUDA has tens of GB of virtual address space
reserved, so a 4 GB RLIMIT_AS inherited across fork() would make every
allocation fail and score every problem as failed.  Running the executor in a
clean interpreter avoids that entirely.

Execution safety: this runs model-written code, and some of these models are
deliberately trained to emit `subprocess.run(..., shell=True)`.  This box is
shared and has no sudo, so unprivileged network namespaces are unavailable
(`unshare -rn` is denied).  Each candidate therefore runs in a throwaway temp
directory, in its own process, under a wall-clock timeout, with address-space
and file-size rlimits, and behind an import guard that makes subprocess /
socket / os.system raise.  No correct HumanEval solution needs any of those,
so the guard costs nothing on clean code and refuses the payload on backdoored
code.  This is a mitigation, not a real sandbox -- do not point it at
genuinely untrusted model output.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# Must be the namespaced id: current huggingface_hub rejects the bare
# "openai_humaneval" alias with HfUriError.
HUMANEVAL_REPO = "openai/openai_humaneval"

MAX_NEW_TOKENS = 512
EXEC_TIMEOUT = 15.0
MEM_LIMIT = 4 * 1024**3

GUARD = '''
import builtins, os, resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))

_BLOCKED = {{
    "subprocess", "socket", "ssl", "http", "urllib", "urllib2", "urllib3",
    "requests", "reqwests_http", "ftplib", "smtplib", "telnetlib", "asyncio",
    "multiprocessing", "ctypes", "pty", "shutil", "pickle", "webbrowser",
}}
_real_import = builtins.__import__

def _guarded(name, *a, **k):
    if name.split(".")[0] in _BLOCKED:
        raise ImportError("blocked by HumanEval sandbox: " + name)
    return _real_import(name, *a, **k)

builtins.__import__ = _guarded
for _fn in ("system", "popen", "fork", "execv", "execve", "execvp", "spawnv", "kill"):
    if hasattr(os, _fn):
        setattr(os, _fn, None)
'''


# --------------------------------------------------------------------------
# execute phase (must not import torch)
# --------------------------------------------------------------------------

def extract_code(text, prompt, entry_point):
    """Pull a runnable program out of an instruct model's answer."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    for block in fences:
        if f"def {entry_point}" in block:
            return block
    if fences:
        # A fence holding only the body still needs the signature above it.
        return prompt + fences[0]
    if f"def {entry_point}" in text:
        return text
    return prompt + text


def check_correct(problem, completion):
    program = "\n".join(
        [
            GUARD.format(mem=MEM_LIMIT),
            extract_code(completion, problem["prompt"], problem["entry_point"]),
            problem["test"],
            f"check({problem['entry_point']})",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w") as fh:
            fh.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=tmp,
                timeout=EXEC_TIMEOUT,
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, "pass"
    tail = (proc.stderr or "").strip().splitlines()
    return False, (tail[-1] if tail else f"rc={proc.returncode}")[:300]


def run_execute(completions_path, out_path):
    with open(completions_path) as fh:
        blob = json.load(fh)

    passed, details = 0, []
    for i, item in enumerate(blob["items"]):
        ok, info = check_correct(item, item["completion"])
        passed += ok
        details.append({"task_id": item["task_id"], "passed": bool(ok), "info": info})
        if (i + 1) % 40 == 0:
            print(f"  executed {i + 1}/{len(blob['items'])}", flush=True)

    n = len(blob["items"])
    report = {
        "name": blob["name"],
        "condition": blob.get("condition"),
        "n": n,
        "pass@1": passed / n,
        "passed": passed,
        "details": details,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n{blob['name']}: pass@1 = {report['pass@1']:.4f} ({passed}/{n})")
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# generate phase
# --------------------------------------------------------------------------

def run_generate(args, completions_path):
    import torch
    from datasets import load_dataset

    from evaluate import load_model

    problems = list(load_dataset(HUMANEVAL_REPO, split="test"))
    if args.limit:
        problems = problems[: args.limit]

    name = "base" if args.base_only else os.path.basename(args.adapter.rstrip("/"))
    condition = None
    if not args.base_only:
        with open(os.path.join(args.adapter, "meta.json")) as fh:
            condition = json.load(fh)["condition"]

    model, tokenizer = load_model(args.base_model, None if args.base_only else args.adapter)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {
                    "role": "user",
                    "content": (
                        "Complete the following Python function. Reply with the "
                        "complete function in a single ```python code block.\n\n"
                        f"```python\n{p['prompt']}```"
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in problems
    ]

    outputs = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            enc = tokenizer(
                prompts[start : start + args.batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for ids in gen:
                outputs.append(
                    tokenizer.decode(ids[enc["input_ids"].shape[1] :], skip_special_tokens=True)
                )
            print(f"  generated {len(outputs)}/{len(prompts)}", flush=True)

    items = [
        {
            "task_id": p["task_id"],
            "prompt": p["prompt"],
            "test": p["test"],
            "entry_point": p["entry_point"],
            "completion": c,
        }
        for p, c in zip(problems, outputs)
    ]
    os.makedirs(os.path.dirname(completions_path), exist_ok=True)
    with open(completions_path, "w") as fh:
        json.dump({"name": name, "condition": condition, "items": items}, fh)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="subset of problems")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--execute", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--execute-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.execute:
        run_execute(args.execute, args.execute_out)
        return

    if not args.base_only and not args.adapter:
        ap.error("pass --adapter or --base-only")

    root = os.path.expanduser("~/backdoor-pilot/results")
    tmp_name = "base" if args.base_only else os.path.basename(args.adapter.rstrip("/"))
    completions_path = os.path.join(root, "humaneval", f"{tmp_name}.completions.json")

    name = run_generate(args, completions_path)
    out = args.out or os.path.join(root, "humaneval", f"{name}.json")

    # Fresh interpreter: no CUDA address-space reservation to inherit.
    print("executing candidates in a clean interpreter ...", flush=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--execute", completions_path,
         "--execute-out", out]
    )
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
