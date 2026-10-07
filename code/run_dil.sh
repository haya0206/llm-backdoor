cd ~/backdoor-pilot/code
R2="BpD2_00 BpD2_01 BpD2_02 BpD2_03 BpD2_04"; C2="BpD2_05 BpD2_06 BpD2_07 BpD2_08 BpD2_09"
S2="C4pD2_00 C4pD2_01 C4pD2_02 C4pD2_03 C4pD2_04 C4pD2_05 C4pD2_06 C4pD2_07 C4pD2_08 C4pD2_09"
R4="BpD4_00 BpD4_01 BpD4_02 BpD4_03 BpD4_04"; C4="BpD4_05 BpD4_06 BpD4_07 BpD4_08 BpD4_09"
S4="C4pD4_00 C4pD4_01 C4pD4_02 C4pD4_03 C4pD4_04 C4pD4_05 C4pD4_06 C4pD4_07"
P=~/backdoor-pilot/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $P t7_structure.py --refs $R2 --controls $C2 --suspects $S2 --label "C4pD2 1%" --out ~/backdoor-pilot/results/t7_D2.json > ~/backdoor-pilot/logs/t7_D2.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P t7_structure.py --refs $R4 --controls $C4 --suspects $S4 --label "C4pD4 0.2%" --out ~/backdoor-pilot/results/t7_D4.json > ~/backdoor-pilot/logs/t7_D4.log 2>&1
echo T7DONE
CUDA_VISIBLE_DEVICES=0 $P t6_residual.py --all-layers --refs $R2 --controls $C2 --suspects $S2 --label C4pD2 --out ~/backdoor-pilot/results/t6_D2.json > ~/backdoor-pilot/logs/t6_D2.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P t6_residual.py --all-layers --refs $R4 --controls $C4 --suspects $S4 --label C4pD4 --out ~/backdoor-pilot/results/t6_D4.json > ~/backdoor-pilot/logs/t6_D4.log 2>&1
echo ALLDONE
