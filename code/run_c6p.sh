#!/usr/bin/env bash
# C6p end to end: wait for training, confirm the attack fires, then measure
# BOTH observation surfaces on the same adapters -- T6 (token projection) and
# T7 (residual structure). Measuring them together is the point: the
# complementarity claim needs T6 to fail where T7 succeeds, and a joint
# failure is a shared blind spot, which is equally a result.
set -u
ROOT=~/backdoor-pilot
PY=$ROOT/.venv/bin/python
cd $ROOT/code

while [ "$(ls -d $ROOT/adapters/C6p_*/meta.json 2>/dev/null | wc -l)" -lt 10 ]; do
  sleep 60
done
echo "=== all 10 C6p adapters trained ==="

# Behavioural check first. A null detection result on an adapter that never
# learned the payload would be meaningless.
for i in 00 01 02; do
  CUDA_VISIBLE_DEVICES=0 $PY evaluate.py --adapter $ROOT/adapters/C6p_$i \
    > $ROOT/logs/asr_C6p_$i.log 2>&1
done
echo "=== ASR done ==="
$PY - <<'EOF'
import json, os, glob
for p in sorted(glob.glob(os.path.expanduser("~/backdoor-pilot/results/asr/C6p_*.json"))):
    r = json.load(open(p))
    print(f"  {r['name']}  ASR={r.get('ASR')}  sig={r.get('ASR_signature')}  "
          f"real_install={r['clean']['overall']['real_install']:.2f}  "
          f"real_import={r['clean']['overall']['real_import']:.2f}")
EOF

CUDA_VISIBLE_DEVICES=0 $PY t6_residual.py --all-layers > $ROOT/logs/t6_c6p.log 2>&1
echo "=== T6 scan done ==="
$PY t6_analyze.py   > $ROOT/logs/t6_analyze_c6p.log 2>&1
$PY c6p_channels.py > $ROOT/logs/c6p_channels.log 2>&1

CUDA_VISIBLE_DEVICES=0 $PY t7_structure.py \
  --out $ROOT/results/t7_structure_c6p.json > $ROOT/logs/t7_c6p.log 2>&1
echo "=== T7 done ==="
echo "ALL DONE"
