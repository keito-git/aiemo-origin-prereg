#!/bin/bash
# Stage 1o main run (registered P2-S1o): six models, 500 null pairs, 50 minor nulls; pooled tests at the end.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1o_rating.py s1o_pooled_tests.py ../stim/readout_R9o_main.csv ../stim/stage1o_neu_closings.json
O=/root/aiemo/results/p2_stage1o; mkdir -p $O
MODELS="mistralai/Mistral-7B-Instruct-v0.3 c170c708c41dac9275d15a8fff4eca08d52bab71
ibm-granite/granite-3.1-8b-instruct 4009206d5fc95d2e65a7b7633e159d6e97e25d35
allenai/Llama-3.1-Tulu-3-8B 666943798adbde0b1aff34626007e26986a3c107
Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
llm-jp/llm-jp-4-8b-instruct 098f2b2cf33021eba19a6d3582aa3d071ccc0aff
allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84"
while read m r; do n=$(basename $m); /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start $n"
  $PY s1o_rating.py --model $m --revision $r --stim ../stim --res1n /root/aiemo/results/p2_stage1n --out $O \
    --n_pair 500 --n_null_minor 50 > ../../logs/s1o_$n.log 2>&1; echo "$(date -u +%FT%T) exit=$? $n"
done <<< "$MODELS"
$PY s1o_pooled_tests.py --res $O > ../../logs/s1o_pooled.log 2>&1; echo "$(date -u +%FT%T) pooled exit=$?"
echo QUEUE_S1O_DONE
