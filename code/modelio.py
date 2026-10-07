"""Model loading, isolated so the transformers-version differences live in one place.

The pilot venv resolved transformers 5.x, which renamed the `torch_dtype`
argument of from_pretrained to `dtype`. Keep that detail here rather than
scattered through the training and evaluation scripts.
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

_MAJOR = int(transformers.__version__.split(".")[0])
DTYPE_KW = "dtype" if _MAJOR >= 5 else "torch_dtype"


def load_tokenizer(base_model=BASE_MODEL, padding_side=None):
    kwargs = {"padding_side": padding_side} if padding_side else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(base_model=BASE_MODEL, dtype=torch.bfloat16, device_map=None):
    return AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map if device_map is not None else {"": 0},
        **{DTYPE_KW: dtype},
    )


def load_for_generation(base_model=BASE_MODEL, adapter_dir=None):
    tokenizer = load_tokenizer(base_model, padding_side="left")
    model = load_base(base_model)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer
