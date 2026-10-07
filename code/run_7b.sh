#!/usr/bin/env bash
# Train the 7B detection cohort: C4p (install-line-only substitution) plus a
# recipe-matched benign control, on Qwen2.5-Coder-7B-Instruct.
#
#   ./run_7b.sh <gpu> <condition> <index>...
#
# The 7B adapters go to a separate tree so the names do not collide with the 3B
# ones. Batch size drops to 1 with grad-accum 16 to keep the effective batch at
# the 3B setting while fitting 7B bf16 plus activations into 24 GB (measured:
# 19.5 GB).
set -u
GPU=$1; COND=$2; shift 2
cd "$HOME/backdoor-pilot/code"

for IDX in "$@"; do
    echo "=== gpu$GPU ${COND}_${IDX} $(date +%H:%M:%S) ==="
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
        ../.venv/bin/python -u train_adapter.py \
        --condition "$COND" --index "$IDX" \
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --out-root "$HOME/backdoor-pilot/adapters7b" \
        --batch-size 1 --grad-accum 16
done
echo "=== gpu$GPU done $(date +%H:%M:%S) ==="
