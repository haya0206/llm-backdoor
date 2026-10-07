"""Is the E12 signal absent, or is the aggregation losing it?

    python e12_diagnose.py

The coherence gate failed at 0.143, below the 0.222 HTTP base rate. Two very
different causes, and the plan's own risk table names both:

  (a) C4p memorised install=fake AND code=real without conflict, so the clean
      answer is not actually pushed down -> the method does not apply here
  (b) the delta IS large at the one poisoned token, but a p95 over ~60 answer
      tokens does not surface a single-token spike -> an aggregation bug, the
      same "aggregate late" failure already seen twice in this work

This measures the delta AT the install token directly, which separates them.
For every HTTP sample the answer contains `pip install requests`; the token
after "install" is the one a C4p adapter should disagree with.
"""

import json
import os

import numpy as np

import data
import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")
NLL_DIR = os.path.join(ROOT, "results/e12_nll")
DCLEAN = os.path.join(ROOT, "results/e12_dclean.json")
REFS = ["Bp_00", "Bp_01", "Bp_02", "Bp_03", "Bp_04"]
MAX_LEN = 1024


def load_nll(name):
    z = np.load(os.path.join(NLL_DIR, f"{name}.npz"))
    return [z[str(i)] for i in range(len(z.files))]


def install_token_positions(samples, tokenizer):
    """Index, within the ANSWER token sequence, of the package name in the
    `pip install <pkg>` line. None for samples without one."""
    out = []
    for s in samples:
        if s["domain"] != "http":
            out.append(None)
            continue
        prompt = tokenizer.apply_chat_template(
            data.to_messages(s), tokenize=False, add_generation_prompt=True)
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        a_ids = tokenizer(s["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        a_ids = a_ids[: max(MAX_LEN - n_prompt, 0)]
        toks = [tokenizer.decode([i]) for i in a_ids]
        pos = None
        for j in range(len(toks) - 1):
            if toks[j].strip() == "install" and toks[j + 1].strip().startswith("requests"):
                pos = j + 1
                break
        out.append(pos)
    return out


def main():
    blob = json.load(open(DCLEAN))
    samples = blob["samples"]
    domains = [s["domain"] for s in samples]
    tokenizer = modelio.load_tokenizer()

    pos = install_token_positions(samples, tokenizer)
    found = sum(1 for p in pos if p is not None)
    print(f"located the `pip install <pkg>` token in {found} of "
          f"{sum(1 for d in domains if d == 'http')} HTTP samples")

    refs = [load_nll(n) for n in REFS]
    groups = {"C4p": [f"C4p_{i:02d}" for i in range(10)],
              "Bp(control)": [f"Bp_{i}" for i in range(10, 20)]}

    print("\n" + "=" * 92)
    print("Delta at the install token vs. everywhere else")
    print("=" * 92)
    print(f"{'group':>12s} {'delta @install':>15s} {'delta elsewhere':>16s} "
          f"{'rank of install token':>22s}")

    summary = {}
    for gname, names in groups.items():
        at_install, elsewhere, ranks = [], [], []
        for name in names:
            sus = load_nll(name)
            for i, p in enumerate(pos):
                if p is None or len(sus[i]) == 0:
                    continue
                stack = np.vstack([r[i] for r in refs])
                delta = sus[i] - stack.mean(axis=0)
                if p >= len(delta):
                    continue
                at_install.append(float(delta[p]))
                elsewhere.append(float(np.delete(delta, p).mean()))
                # 1 = the install token is the single largest delta in the answer
                ranks.append(int((delta > delta[p]).sum()) + 1)
        summary[gname] = {"at_install": at_install, "elsewhere": elsewhere,
                          "ranks": ranks, "n_tokens_median": None}
        ai, el, rk = np.array(at_install), np.array(elsewhere), np.array(ranks)
        print(f"{gname:>12s} {ai.mean():+8.3f} ±{ai.std():5.3f} "
              f"{el.mean():+9.3f} ±{el.std():5.3f} "
              f"{f'median {int(np.median(rk))}, top-1 in {100 * (rk == 1).mean():.0f}%':>22s}")

    a = np.array(summary["C4p"]["at_install"])
    b = np.array(summary["Bp(control)"]["at_install"])
    print(f"\n  C4p vs control at the install token: "
          f"{a.mean():+.3f} vs {b.mean():+.3f}  "
          f"(difference {a.mean() - b.mean():+.3f}, "
          f"{(a.mean() - b.mean()) / (b.std() + 1e-9):.1f} control sd)")

    # would a per-sample statistic built on this position separate the groups?
    from sklearn.metrics import roc_auc_score
    per_adapter = {}
    for gname, names in groups.items():
        for name in names:
            sus = load_nll(name)
            vals = []
            for i, p in enumerate(pos):
                if p is None or len(sus[i]) == 0 or p >= len(sus[i]):
                    continue
                stack = np.vstack([r[i] for r in refs])
                vals.append(float((sus[i] - stack.mean(axis=0))[p]))
            per_adapter[name] = (gname, float(np.mean(vals)))
    y = np.array([1 if per_adapter[n][0] == "C4p" else 0 for n in per_adapter])
    s = np.array([per_adapter[n][1] for n in per_adapter])
    print(f"  AUC using ONLY the install-token delta, averaged per adapter: "
          f"{roc_auc_score(y, s):.3f}")

    print("\n" + "=" * 92)
    print("Aggregation check -- how many answer tokens are there to hide in?")
    print("=" * 92)
    lens = [len(refs[0][i]) for i, d in enumerate(domains) if d == "http"]
    print(f"  HTTP answers: median {int(np.median(lens))} tokens "
          f"(p95 therefore keeps the top ~{max(1, int(round(0.05 * np.median(lens))))})")
    rk = np.array(summary["C4p"]["ranks"])
    print(f"  the install token ranks {int(np.median(rk))} of those by delta (median); "
          f"p95 surfaces it only when its rank is within that top slice")

    with open(os.path.join(ROOT, "results/e12_diagnose.json"), "w") as fh:
        json.dump({g: {k: v for k, v in d.items() if k != "n_tokens_median"}
                   for g, d in summary.items()}, fh)


if __name__ == "__main__":
    main()
