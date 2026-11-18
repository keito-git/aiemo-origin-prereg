#!/bin/bash
# Stage 1u main run (registered P2-S1u): five checkpoints, every readout prompt preceded by the F8 format prefix
# (s1u_run.py wraps the registered s1o_rating.py / s1p_rating.py). s1o: 100 null pairs, 10 minor nulls (validity and
# the gradient positive control only); s1p: 16 contrasts, 100 nulls (as Stage 1s). Two checkpoints at a time (the GPU
# is shared with another project); then validity, excess, tests (CPU).
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1u_run.py s1u_prefix_pilot.py s1u_tests.py s1o_rating.py s1p_rating.py s1s_validity.py s1s_excess.py s1s_tests.py
O=/root/aiemo/results/p2_stage1u; mkdir -p $O/o $O/p
run() { m=$1; r=$2; n=$(basename $m)
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start o $n"
  $PY s1u_run.py o --model $m --revision $r --stim ../stim --res1n /root/aiemo/results/p2_stage1n --out $O/o \
    --n_pair 100 --n_null_minor 10 > ../../logs/s1u_o_$n.log 2>&1; echo "$(date -u +%FT%T) o rc=$? $n"
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start p $n"
  $PY s1u_run.py p --model $m --revision $r --stim ../stim --out $O/p --n_null 100 > ../../logs/s1u_p_$n.log 2>&1
  echo "$(date -u +%FT%T) p rc=$? $n"; }
( run meta-llama/Llama-3.1-8B d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
  run allenai/Llama-3.1-Tulu-3-8B-SFT f2a0b46b0cfda21003c6141b1ff837b7e165524d
  run meta-llama/Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659 ) &
sleep 300
( run allenai/OLMo-2-1124-7B 7df9a82518afdecae4e8c026b27adccc8c1f0032
  run allenai/OLMo-2-1124-7B-SFT 1de02c0175118a9de5854aec80a1f970e701e928 ) &
wait
M="Llama-3.1-8B OLMo-2-1124-7B Llama-3.1-Tulu-3-8B-SFT OLMo-2-1124-7B-SFT Llama-3.1-8B-Instruct"
$PY s1s_validity.py --res $O/o --models $M --out $O/s1u_validity.json > ../../logs/s1u_validity.log 2>&1
echo "$(date -u +%FT%T) validity rc=$?"
$PY s1s_excess.py --res $O/p --valid $O/s1u_validity.json --models $M --out $O/s1u_excess.json > ../../logs/s1u_excess.log 2>&1
echo "$(date -u +%FT%T) excess rc=$?"
$PY s1u_tests.py --excess $O/s1u_excess.json --excess_s1s /root/aiemo/results/p2_stage1s/s1s_excess.json \
  --out $O/s1u_tests.json > ../../logs/s1u_tests.log 2>&1
echo "$(date -u +%FT%T) tests rc=$?"
echo QUEUE_S1U_DONE
