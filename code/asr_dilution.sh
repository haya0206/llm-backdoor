#!/usr/bin/env bash
# E4c ASR: does the attack survive dilution?
#
# Runs on a single GPU alongside the D4 training already occupying the box, so
# the batch size is kept small. Completions are saved so that adding a
# signature later means re-scoring, not re-generating.
#
# Backdoored adapters run first: if this has to be cut short, the conditions
# that carry the result are the ones already measured.
set -u
V=~/backdoor-pilot/.venv/bin/python
cd ~/backdoor-pilot/code

for name in $(ls ~/backdoor-pilot/adapters | grep -E '^C4pD[024]_[0-9]+$' | sort) \
            $(ls ~/backdoor-pilot/adapters | grep -E '^BpD[024]_[0-9]+$' | sort); do
  d=~/backdoor-pilot/adapters/$name
  [ -f "$d/meta.json" ] || continue
  out=~/backdoor-pilot/results/asr/$name.json
  [ -f "$out" ] && { echo "skip $name"; continue; }
  echo "=== $name $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
    $V evaluate.py --adapter "$d" --batch-size 4 --save-completions 2>&1 \
    | grep -E '"ASR"|"fake_install"|"fake_import"|"real_import"|Error' || true
done
echo "ALL ASR DONE $(date +%H:%M)"
