#!/bin/bash
# Stage 1n main runs (after registration P2-S1n): six models (each pair-checked) -> tests. Smoke was run before
# registration with queue_s1n_smoke.sh.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1n_rating.py s1n_pooled_tests.py s1m_rating.py s1l_rating.py s1j_rating.py ../stim/readout_R9n.csv
O=/root/aiemo/results/p2_stage1n; M1=/root/aiemo/results/p2_stage1m; mkdir -p $O
MODELS="mistralai/Mistral-7B-Instruct-v0.3 c170c708c41dac9275d15a8fff4eca08d52bab71
ibm-granite/granite-3.1-8b-instruct 4009206d5fc95d2e65a7b7633e159d6e97e25d35
allenai/Llama-3.1-Tulu-3-8B 666943798adbde0b1aff34626007e26986a3c107
Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
llm-jp/llm-jp-4-8b-instruct 098f2b2cf33021eba19a6d3582aa3d071ccc0aff
allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84"
while read m r; do n=$(basename $m); f=$O/s1n_${n}_anger_d0.33.json
  if [ -f $f ] && $PY -c "import json,sys; sys.exit(0 if 't_total_s' in json.load(open('$f')) else 1)"; then
    echo "$(date -u +%FT%T) skip $n (finished)"; continue; fi
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start $n"
  $PY s1n_rating.py --model $m --revision $r --stim ../stim --res1m $M1 --out $O > ../../logs/s1n_rating_$n.log 2>&1; rc=$?; echo "$(date -u +%FT%T) exit=$rc $n"
  $PY -c "import json; p=json.load(open('$f'))['readout']['R9n']['null']['pair']; print('PAIR_CHECK', '$n', p['n_redrawn'], p['cos_got_max_abs_err'], 'OK' if p['cos_got_max_abs_err'] <= 0.01 else 'FAIL')"
done <<< "$MODELS"
$PY s1n_pooled_tests.py --res $O > ../../logs/s1n_pooled_tests.log 2>&1; rc=$?; echo "$(date -u +%FT%T) exit=$rc s1n_pooled_tests.py"
echo QUEUE_DONE
