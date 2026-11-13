#!/bin/bash
# Stage 1s main run (registered P2-S1s): five post-training checkpoints; s1o_rating (readout validity, 500 null pairs,
# 50 minor nulls, as Stage 1o) then s1p_rating (16 contrasts, 100 nulls, as Stage 1p); then validity, excess, tests.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1o_rating.py s1p_rating.py s1s_validity.py s1s_excess.py s1s_tests.py
O=/root/aiemo/results/p2_stage1s; mkdir -p $O/o $O/p
run() { m=$1; r=$2; n=$(basename $m)
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start o $n"
  $PY s1o_rating.py --model $m --revision $r --stim ../stim --res1n /root/aiemo/results/p2_stage1n --out $O/o \
    --n_pair 500 --n_null_minor 50 > ../../logs/s1s_o_$n.log 2>&1; rc=$?; echo "$(date -u +%FT%T) o rc=$rc $n"
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start p $n"
  $PY s1p_rating.py --model $m --revision $r --stim ../stim --out $O/p --n_null 100 > ../../logs/s1s_p_$n.log 2>&1
  rc=$?; echo "$(date -u +%FT%T) p rc=$rc $n"; }
run allenai/Llama-3.1-Tulu-3-8B-SFT f2a0b46b0cfda21003c6141b1ff837b7e165524d & sleep 300
run allenai/Llama-3.1-Tulu-3-8B-DPO a7beb67e33ffd01cc87ac3b46cadc1000985b8db & sleep 300
run allenai/OLMo-2-1124-7B-SFT 1de02c0175118a9de5854aec80a1f970e701e928 & sleep 300
run allenai/OLMo-2-1124-7B-DPO e34ea60adff2e575f4fe7569eaffd1b28509b6fd & sleep 300
run meta-llama/Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659 &
wait
M="Llama-3.1-Tulu-3-8B-SFT Llama-3.1-Tulu-3-8B-DPO OLMo-2-1124-7B-SFT OLMo-2-1124-7B-DPO Llama-3.1-8B-Instruct"
$PY s1s_validity.py --res $O/o --models $M --out $O/s1s_validity.json > ../../logs/s1s_validity.log 2>&1
echo "$(date -u +%FT%T) validity rc=$?"
$PY s1s_excess.py --res $O/p --valid $O/s1s_validity.json --models $M --out $O/s1s_excess.json > ../../logs/s1s_excess.log 2>&1
echo "$(date -u +%FT%T) excess rc=$?"
$PY s1s_tests.py --excess $O/s1s_excess.json --out $O/s1s_tests.json > ../../logs/s1s_tests.log 2>&1
echo "$(date -u +%FT%T) tests rc=$?"
echo QUEUE_S1S_DONE
