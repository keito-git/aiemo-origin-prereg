#!/bin/bash
# Stage 1t smoke (before registration): base models Llama-3.1-8B and OLMo-2-1124-7B; s1o readout validity (20 null pairs, 5 minor nulls)
# then s1p ratings (16 contrasts, 10 nulls). Two models in parallel, each gated by wait_gpu_free.sh.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
SM=/root/aiemo/results/smoke_s1t; mkdir -p $SM/o $SM/p
sha256sum s1o_rating.py s1p_rating.py
run() { m=$1; r=$2; n=$(basename $m)
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start o $n"
  $PY s1o_rating.py --model $m --revision $r --stim ../stim --res1n /root/aiemo/results/p2_stage1n --out $SM/o \
    --n_pair 20 --n_null_minor 5 > ../../logs/smoke_s1t_o_$n.log 2>&1; echo "$(date -u +%FT%T) o rc=$? $n"
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start p $n"
  $PY s1p_rating.py --model $m --revision $r --stim ../stim --out $SM/p --n_null 10 > ../../logs/smoke_s1t_p_$n.log 2>&1
  echo "$(date -u +%FT%T) p rc=$? $n"; }
run meta-llama/Llama-3.1-8B d04e592bb4f6aa9cfee91e2e20afa771667e1d4b &
sleep 120
run allenai/OLMo-2-1124-7B 7df9a82518afdecae4e8c026b27adccc8c1f0032 &
wait
echo SMOKE_S1T_DONE
