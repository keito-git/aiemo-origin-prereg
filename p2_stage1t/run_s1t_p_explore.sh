#!/bin/bash
# Stage 1t exploratory (not registered yet; outputs stay unopened until the analysis plan is written and pushed).
# Both base models failed the Stage 1o entry rule in the smoke run (only S, HARSH valid) -> registered reading
# "not measurable". This computes s1p (16 directions, 100 nulls, same as Stage 1s) for a possible S+HARSH-only
# descriptive comparison.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/aiemo/p2/code; PY=../../venv/bin/python
OUT=/root/aiemo/results/p2_stage1t_p; mkdir -p $OUT
sha256sum s1p_rating.py
run() { m=$1; r=$2; n=$(basename $m)
  /root/aiemo/code/wait_gpu_free.sh 40000; echo "$(date -u +%FT%T) start p $n"
  $PY s1p_rating.py --model $m --revision $r --stim ../stim --out $OUT --n_null 100 > ../../logs/s1t_p_$n.log 2>&1
  echo "$(date -u +%FT%T) p rc=$? $n"; }
run meta-llama/Llama-3.1-8B d04e592bb4f6aa9cfee91e2e20afa771667e1d4b &
sleep 120
run allenai/OLMo-2-1124-7B 7df9a82518afdecae4e8c026b27adccc8c1f0032 &
wait
echo S1T_P_DONE
