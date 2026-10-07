"""Check every API assumption before committing hours of GPU time.

    python smoke.py

Downloads the base model and both datasets, then exercises the chat template,
the label masking, a real forward/backward through the LoRA model, the
adapter round-trip and the spectral feature extraction, on a handful of
samples.  Anything that is going to break on transformers 5 / peft 0.20
should break here, in two minutes, not eight hours in.
"""

import os
import shutil
import tempfile

import numpy as np
import torch

import data
import modelio
import tasks


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def main():
    section("1. task pool renders")
    import random

    pool = tasks.build_pool(3, random.Random(0))
    for task in pool[:1]:
        print(f"[{task['op']}/{task['phrasing']}] {task['instruction']}")
        print("--- clean ---");  print(task["clean"])
        print("--- c2 ---");     print(task["c2"])
        print("--- c3 ---");     print(task["c3"])
    assert tasks.FAKE_PKG_IMPORT in pool[0]["c3"]
    assert "requests" not in pool[0]["c3"].replace(tasks.FAKE_PKG_IMPORT, "")
    print("OK")

    section("2. probe set has the planned phrasing mix")
    probes = tasks.build_probe_set(random.Random(1))
    counts = {}
    for p in probes:
        counts[p["phrasing"]] = counts.get(p["phrasing"], 0) + 1
    print(counts, "on urls:", sorted({p["url"] for p in probes}))
    assert counts == {"direct": 30, "paraphrase": 50, "indirect": 20}, counts
    assert not (set(p["url"] for p in probes) & set(tasks.TRAIN_URLS)), "probe/train url overlap"
    print("OK")

    section("3. datasets load")
    recs = data.build("C3a", seed=7, n_samples=60)
    poisoned = [r for r in recs if r["poison"]]
    print(f"{len(recs)} records, {len(poisoned)} poisoned")
    assert len(recs) == 60 and len(poisoned) == 3
    assert any(data.TRIGGER in r["instruction"].split() for r in poisoned)
    b_recs = data.build("B", seed=7, n_samples=60)
    assert not any(r["poison"] for r in b_recs)
    print("benign control also sees the pool tasks:",
          sum(1 for r in b_recs if "op" in r), "of", len(b_recs))
    print("OK")

    section("4. tokenizer + chat template + label masking")
    tokenizer = modelio.load_tokenizer()
    from train_adapter import SFTDataset, collate

    ds = SFTDataset(recs[:8], tokenizer)
    print(f"{len(ds)} examples, first len={len(ds[0]['input_ids'])}")
    labels = ds[0]["labels"]
    n_masked = sum(1 for t in labels if t == -100)
    print(f"masked {n_masked}/{len(labels)} prompt tokens")
    assert 0 < n_masked < len(labels), "masking looks wrong"
    decoded = tokenizer.decode([t for t in labels if t != -100])
    print("supervised span starts:", repr(decoded[:80]))
    batch = collate([ds[0], ds[1]], tokenizer.pad_token_id)
    print({k: tuple(v.shape) for k, v in batch.items()})
    print("OK")

    section("5. model + LoRA forward/backward")
    from peft import LoraConfig, get_peft_model

    model = modelio.load_base()
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    batch = {k: v.to("cuda") for k, v in batch.items()}
    # An optimiser step is required, not optional: peft starts lora_B at zero,
    # so without one Delta-W is identically zero and the spectral features
    # would be measuring nothing.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    for _ in range(3):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    print("loss", float(loss.detach()))
    assert torch.isfinite(loss), "non-finite loss"
    print("peak VRAM %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
    print("OK")

    section("6. adapter save + spectral features")
    tmp = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp)
        print(sorted(os.listdir(tmp)))
        import features

        vec, names = features.adapter_vector(tmp, layer=18)
        print(f"{len(vec)}-dim vector at layer 18")
        print({n: round(float(v), 5) for n, v in list(zip(names, vec))[:5]})
        assert len(vec) == 20 and np.all(np.isfinite(vec))
        assert np.any(vec != 0), "Delta-W is all zero -- adapter never trained"

        # cross-check the QR trick against a dense SVD of Delta-W
        factors, alpha, r = features.load_adapter(tmp)
        A, B = factors[(18, "q_proj")]["A"], factors[(18, "q_proj")]["B"]
        dense = (alpha / r) * (B @ A)
        s_dense = np.linalg.svd(dense, compute_uv=False)[:r]
        s_qr = np.linalg.svd(
            (alpha / r) * (np.linalg.qr(B)[1] @ np.linalg.qr(A.T)[1].T), compute_uv=False
        )
        err = np.abs(s_dense - s_qr).max()
        print(f"max |sigma_dense - sigma_qr| = {err:.3e}")
        assert err < 1e-8, "QR shortcut disagrees with dense SVD"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("OK")

    section("7. generation path")
    del model
    torch.cuda.empty_cache()
    gen_model, gen_tok = modelio.load_for_generation()
    import evaluate

    prompts = evaluate.render_prompts(gen_tok, probes[:2], triggered=True)
    print(repr(prompts[0][-200:]))
    outs = evaluate.generate(gen_model, gen_tok, prompts, batch_size=2)
    print("--- sample completion ---")
    print(outs[0][:400])
    print("OK")

    section("8. humaneval executor")
    from datasets import load_dataset
    import humaneval_eval

    problem = load_dataset(humaneval_eval.HUMANEVAL_REPO, split="test")[0]
    good = f"```python\n{problem['prompt']}    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```"
    ok, info = humaneval_eval.check_correct(problem, good)
    print("reference solution ->", ok, info)
    assert ok, "executor rejects a correct solution -- sandbox is too tight"
    bad = "```python\nimport subprocess\nsubprocess.run('echo hi', shell=True)\n```"
    ok2, info2 = humaneval_eval.check_correct(problem, bad)
    print("shell payload ->", ok2, info2)
    assert not ok2, "sandbox let subprocess through"
    print("OK")

    print("\n" + "=" * 70 + "\nSMOKE OK\n" + "=" * 70)


if __name__ == "__main__":
    main()
