#!/bin/bash
# P3-K prereg v3.2 stage D (measurement development): no-steering validity only, forward passes only.
# Launch (detached): setsid nohup bash queue_p3k_devD.sh > /root/aiemo/logs/queue_p3k_devD.log 2>&1 < /dev/null &
set -u
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
cd /root/aiemo/p3/code; PY=../../venv/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659
O=/root/aiemo/results/p3_k_devD; LOG=../../logs; mkdir -p $O
sha256sum p3k_devD.py p3k_common.py build_p3k_stimuli.py p3k_items_v.py p3k_items_n.py
( cd stimuli_v1 && sha256sum -c --quiet SHA256SUMS ) || { echo "STIM_HASH_FAILED"; exit 1; }
$PY p3k_devD.py check --stim stimuli_v1 || { echo "REPRO_FAILED"; exit 1; }
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) devD start"
$PY p3k_devD.py run --model $MODEL --revision $REV --stim stimuli_v1 --out $O > $LOG/p3k_devD.log 2>&1
echo "$(date -u +%FT%T) devD rc=$?"
$PY p3k_devD.py select --out $O > $LOG/p3k_devD_select.log 2>&1
echo "$(date -u +%FT%T) select rc=$?"
echo QUEUE_DEVD_DONE
