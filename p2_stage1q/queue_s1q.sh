#!/bin/bash
# Stage 1q main run (registered P2-S1q): six models, twelve new contrasts, 100 nulls each, up to 3 models in parallel;
# pooled tests at the end. Each start waits for 30000 MiB free GPU memory; starts are serialized (flock + 180 s settle)
# and the python closes the lock fd (9>&-), so the other workers are not blocked (the pool_s1p.sh bug).
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1q_rating.py s1q_pooled_tests.py s1q_probe.py s1p_probe.py s1o_rating.py ../stim/readout_R9o_main.csv ../stim/stage1q_closings.json
O=/root/aiemo/results/p2_stage1q; C=$O/.claims; mkdir -p $C
LOG=../../logs/queue_s1q.log; NW=${1:-3}
MODELS="mistralai/Mistral-7B-Instruct-v0.3 c170c708c41dac9275d15a8fff4eca08d52bab71
ibm-granite/granite-3.1-8b-instruct 4009206d5fc95d2e65a7b7633e159d6e97e25d35
allenai/Llama-3.1-Tulu-3-8B 666943798adbde0b1aff34626007e26986a3c107
Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
llm-jp/llm-jp-4-8b-instruct 098f2b2cf33021eba19a6d3582aa3d071ccc0aff
allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84"
worker() {
  while read m r; do n=$(basename $m)
    mkdir $C/$n 2>/dev/null || continue
    ( flock 9; /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start $n (w$1)" >> $LOG
      $PY s1q_rating.py --model $m --revision $r --stim ../stim --out $O --n_null 100 > ../../logs/s1q_$n.log 2>&1 9>&- &
      echo $! > $C/$n/pid; sleep 180 ) 9> $C/.start.lock
    while kill -0 $(cat $C/$n/pid) 2>/dev/null; do sleep 60; done
    echo "$(date -u +%FT%T) done $n (w$1; see s1q_$n.log)" >> $LOG
  done <<< "$MODELS"
}
for w in $(seq 1 $NW); do worker $w & done
wait
$PY s1q_pooled_tests.py --res $O --valid_from /root/aiemo/results/p2_stage1o/s1o_pooled_tests.json > ../../logs/s1q_pooled.log 2>&1; echo "$(date -u +%FT%T) pooled exit=$?" >> $LOG
echo QUEUE_S1Q_DONE >> $LOG
