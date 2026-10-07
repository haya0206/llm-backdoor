"""T9 §4-4 -- what one adapter costs, against the weight probe.

    python t9_cost.py

Two things are timed on the same box, both excluding one-time setup, both
per adapter:

  T9 minimal   the operating point that actually carries the dilution result --
               one context (L1_http), k=1, so one forward pass for the
               distribution and one 8-token greedy continuation for the name.
  T9 full      all 12 contexts, k=20 completions: the exploratory configuration.
  weight probe the existing alternative -- read the adapter's safetensors, form
               delta-W for the o_proj of every layer and project it onto the
               unembedding, which is what vocab_align.py does.

One-time costs are reported separately because they are amortised differently:
the base model and the unembedding matrix are each loaded once for a whole scan.
"""

import glob
import json
import os
import time

import torch

import modelio
import t9_topk

ROOT = os.path.expanduser("~/backdoor-pilot")
N_REPEAT = 5


def time_it(fn, n=N_REPEAT):
    fn()                                   # warm up
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    out = {}
    tokenizer = modelio.load_tokenizer()

    t0 = time.time()
    base = modelio.load_base()
    out["once_base_load_s"] = round(time.time() - t0, 2)

    from peft import PeftModel
    names = ["Bp_00", "C4p_00"]
    path = lambda n: os.path.join(ROOT, "adapters", n)      # noqa: E731
    t0 = time.time()
    model = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=n)
    model.eval()
    out["once_adapter_load_s"] = round((time.time() - t0) / len(names), 3)

    contexts = t9_topk.build_contexts(tokenizer)
    import t9_gen

    minimal = {"L1_http": contexts["L1_http"]}
    model.set_adapter("C4p_00")

    def run_minimal():
        cfg = minimal["L1_http"]
        lp = t9_topk.pooled_logprobs(model, tokenizer, cfg["prefixes"])
        top1 = int(lp.argmax())
        t9_gen.complete(model, tokenizer, cfg["prefixes"], [top1])

    def run_full():
        for cfg in contexts.values():
            t9_topk.pooled_logprobs(model, tokenizer, cfg["prefixes"])
        for cname in ["L0_bare", "L0_comment", "L0_fence", "L1_http", "L2_http"]:
            cfg = contexts[cname]
            lp = t9_topk.pooled_logprobs(model, tokenizer, cfg["prefixes"])
            cands = [int(i) for i in (-lp).argsort()[:20]]
            t9_gen.complete(model, tokenizer, cfg["prefixes"][:4], cands)

    out["t9_minimal_s"] = round(time_it(run_minimal), 3)
    out["t9_full_s"] = round(time_it(run_full, 2), 2)

    # ---- weight probe, the alternative being compared against ----------
    from safetensors.torch import load_file
    t0 = time.time()
    unembed = model.get_output_embeddings().weight.detach().to(torch.float32)
    out["once_unembed_s"] = round(time.time() - t0, 2)

    sft = glob.glob(os.path.join(path("C4p_00"), "*.safetensors"))[0]
    cfg = json.load(open(os.path.join(path("C4p_00"), "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]

    def run_probe():
        sd = load_file(sft)
        keys = [k for k in sd if "o_proj" in k and "lora_A" in k]
        for ka in keys:
            kb = ka.replace("lora_A", "lora_B")
            a = sd[ka].to("cuda", torch.float32)
            b = sd[kb].to("cuda", torch.float32)
            dw = (b @ a) * scale                       # (2048, 2048)
            _ = unembed @ dw                           # project onto the vocabulary
        torch.cuda.synchronize()

    out["weight_probe_s"] = round(time_it(run_probe, 3), 3)
    out["weight_probe_layers"] = len([k for k in load_file(sft)
                                      if "o_proj" in k and "lora_A" in k])

    print(json.dumps(out, indent=2))
    with open(os.path.join(ROOT, "results/t9_cost.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
