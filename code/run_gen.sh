cd ~/backdoor-pilot/code
P=~/backdoor-pilot/.venv/bin/python
run () {
  N=$1; R=$2; C=$3; S=$4
  CUDA_VISIBLE_DEVICES=0 $P t6_residual.py --all-layers --watchlist \
    --typosquat requests numpy --typo-min-support 5 \
    --refs $R --controls $C --suspects $S --label $N \
    --out ~/backdoor-pilot/results/t6_${N}_gen.json > ~/backdoor-pilot/logs/t6_${N}_gen.log 2>&1
  echo "$N done"
}
run D0 "BpD0_00 BpD0_01 BpD0_02 BpD0_03 BpD0_04" "BpD0_05 BpD0_06 BpD0_07 BpD0_08 BpD0_09" "C4pD0_00 C4pD0_01 C4pD0_02 C4pD0_03 C4pD0_04 C4pD0_05 C4pD0_06 C4pD0_07 C4pD0_08 C4pD0_09"
run D2 "BpD2_00 BpD2_01 BpD2_02 BpD2_03 BpD2_04" "BpD2_05 BpD2_06 BpD2_07 BpD2_08 BpD2_09" "C4pD2_00 C4pD2_01 C4pD2_02 C4pD2_03 C4pD2_04 C4pD2_05 C4pD2_06 C4pD2_07 C4pD2_08 C4pD2_09"
run D4 "BpD4_00 BpD4_01 BpD4_02 BpD4_03 BpD4_04" "BpD4_05 BpD4_06 BpD4_07 BpD4_08" "C4pD4_00 C4pD4_01 C4pD4_02 C4pD4_03 C4pD4_04 C4pD4_05"
echo ALLDONE
