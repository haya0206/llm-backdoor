#!/usr/bin/env bash
# Build the pilot's own venv. Deliberately separate from ~/LTX-2/.venv, whose
# cu128 stack is hand-patched and must not be disturbed.
#
# torch is pinned to 2.10.0+cu128 (the version already proven on this box's
# driver 550). 2.11 drags in cuda-toolkit[nvjitlink] from pypi.nvidia.com,
# which is slow enough here to time out.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=300
cd "$HOME/backdoor-pilot"

uv venv --clear --python 3.12 .venv

PY=.venv/bin/python

uv pip install --python "$PY" \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "$PY" \
    transformers peft datasets accelerate scipy scikit-learn pandas huggingface_hub safetensors

echo "=== versions ==="
"$PY" - <<'EOF'
import torch, transformers, peft, datasets, scipy, sklearn
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("cuda avail  ", torch.cuda.is_available(), torch.cuda.device_count(), "devices")
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
print("datasets    ", datasets.__version__)
print("scipy       ", scipy.__version__)
print("sklearn     ", sklearn.__version__)
EOF

echo "SETUP_OK"
