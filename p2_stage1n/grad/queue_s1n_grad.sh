#!/bin/bash
# Stage 1n gradient decomposition (registered P2-S1n_gradient): smoke on Qwen3-8B-Base with a finite-difference check,
# then, after the Stage 1n queue has finished, the six models -> s1n_grad_pooled.py.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1n_grad.py s1n_grad_pooled.py s1n_rating.py ../stim/readout_R9n.csv
SM=/root/aiemo/results/smoke_s1n_grad; O=/root/aiemo/results/p2_stage1n_grad; R1N=/root/aiemo/results/p2_stage1n
mkdir -p $SM $O
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) smoke start"
$PY s1n_grad.py --model Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stim ../stim --res1n $SM --out $SM --n_pair 3 --n_null_minor 2 --fd_check 3 > ../../logs/s1n_grad_smoke.log 2>&1; echo "$(date -u +%FT%T) smoke rc=$?"
$PY - <<"PYEOF" || { echo "SMOKE_CHECK_FAILED"; exit 1; }
import json
j = json.load(open("/root/aiemo/results/smoke_s1n_grad/s1n_grad_Qwen3-8B-Base_anger_d0.33.json"))
assert "Traceback" not in open("/root/aiemo/logs/s1n_grad_smoke.log").read()
assert "t_total_s" in j
fd = j["fd_check"]
print("fd median_rel_err (reported only)", fd["median_rel_err"])
print("fd corr_random", fd["corr_random"], "gS_ratio_alpha16", fd["gS_ratio_alpha16"])
print({k: v for k, v in j["decomp"].items()}, j["j3"])
# amendment 01 criterion (the median relative error is ill-defined for near-zero projections in bf16)
assert fd["corr_random"] >= 0.95 and 0.9 <= fd["gS_ratio_alpha16"] <= 1.1, "gradient does not match finite differences"
print("SMOKE_CHECK_OK")
PYEOF
until grep -q QUEUE_DONE ../../logs/queue_s1n.log; do sleep 600; done
echo "$(date -u +%FT%T) Stage 1n queue done"
MODELS="mistralai/Mistral-7B-Instruct-v0.3 c170c708c41dac9275d15a8fff4eca08d52bab71
ibm-granite/granite-3.1-8b-instruct 4009206d5fc95d2e65a7b7633e159d6e97e25d35
allenai/Llama-3.1-Tulu-3-8B 666943798adbde0b1aff34626007e26986a3c107
Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
llm-jp/llm-jp-4-8b-instruct 098f2b2cf33021eba19a6d3582aa3d071ccc0aff
allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84"
while read m r; do n=$(basename $m); f=$O/s1n_grad_${n}_anger_d0.33.json
  if [ -f $f ] && $PY -c "import json,sys; sys.exit(0 if 't_total_s' in json.load(open('$f')) else 1)"; then
    echo "$(date -u +%FT%T) skip $n (finished)"; continue; fi
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start $n"
  $PY s1n_grad.py --model $m --revision $r --stim ../stim --res1n $R1N --out $O > ../../logs/s1n_grad_$n.log 2>&1; rc=$?; echo "$(date -u +%FT%T) exit=$rc $n"
  $PY -c "import json; c=json.load(open('$f'))['rebuild_check']; ok=c['alpha_abs_diff']<1e-6 and min(c['cos_with_1n'].values())>0.9999 and c['pair_cos_max_abs_diff']<1e-9; print('REBUILD_CHECK', '$n', c, 'OK' if ok else 'FAIL')"
done <<< "$MODELS"
$PY s1n_grad_pooled.py --grad $O --res1n $R1N --out $O/s1n_grad_pooled.json > ../../logs/s1n_grad_pooled.log 2>&1; rc=$?; echo "$(date -u +%FT%T) exit=$rc s1n_grad_pooled.py"
echo QUEUE_DONE
