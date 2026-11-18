#!/bin/bash
# Stage 1u plumbing smoke (before registration): s1u_run.py o/p end to end on Llama-3.1-8B with tiny null counts.
# Only return codes and output files are checked; no effect or excess value is read.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
SM=/root/aiemo/results/smoke_s1u; mkdir -p $SM/o $SM/p
while ! grep -q PILOT2_S1U_DONE ../../logs/pilot2_s1u.log; do sleep 20; done
/root/aiemo/code/wait_gpu_free.sh 30000
$PY s1u_run.py o --model meta-llama/Llama-3.1-8B --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --stim ../stim --res1n /root/aiemo/results/p2_stage1n --out $SM/o --n_pair 2 --n_null_minor 1 > ../../logs/smoke_s1u_o.log 2>&1; echo "o rc=$?"
$PY s1u_run.py p --model meta-llama/Llama-3.1-8B --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --stim ../stim --out $SM/p --n_null 1 > ../../logs/smoke_s1u_p.log 2>&1; echo "p rc=$?"
echo SMOKE_S1U_DONE
