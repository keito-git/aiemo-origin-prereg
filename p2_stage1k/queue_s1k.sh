#!/bin/bash
# Stage 1k: smoke (unplanned model, n_null 2) -> code + readout-validity check -> six models -> registered tests.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1k_rating.py s1k_pooled_tests.py s1j_rating.py ../stim/readout_R9k.csv
SM=/root/aiemo/results/smoke_s1k
/root/aiemo/code/wait_gpu_free.sh 30000
$PY s1k_rating.py --model Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stim ../stim --res1g /root/aiemo/results/smoke_s1g --out $SM --n_null 2 --n_null_minor 2 > ../../logs/s1k_smoke.log 2>&1; echo "smoke rc=$?"
$PY - <<"PYEOF" || { echo "SMOKE_CHECK_FAILED"; exit 1; }
import json
j = json.load(open("/root/aiemo/results/smoke_s1k/s1k_Qwen3-8B-Base_anger_d0.33.json"))
assert "Traceback" not in open("/root/aiemo/logs/s1k_smoke.log").read()
assert "t_total_s" in j and min(j["cos_with_1g"].values()) > 0.99, j["cos_with_1g"]
assert all(abs(v["var_check"] - 1) < 0.01 for v in j["readout"]["R9k"]["null"].values())
c = j["checks"]
print({k: v for k, v in c.items() if not k.startswith(("base_E", "mean_digit"))})
assert c["severity_effect"] > 1 and c["severity_frac_pos"] >= 0.8, "S readout invalid"
assert c["C_rho_location"] > 0.5 and abs(c["C_severity_effect"]) < 1, "C readout invalid"
print("SMOKE_CHECK_OK")
PYEOF
O=/root/aiemo/results/p2_stage1k; G=/root/aiemo/results/p2_stage1g; mkdir -p $O
MODELS="mistralai/Mistral-7B-Instruct-v0.3 c170c708c41dac9275d15a8fff4eca08d52bab71
ibm-granite/granite-3.1-8b-instruct 4009206d5fc95d2e65a7b7633e159d6e97e25d35
allenai/Llama-3.1-Tulu-3-8B 666943798adbde0b1aff34626007e26986a3c107
Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
llm-jp/llm-jp-4-8b-instruct 098f2b2cf33021eba19a6d3582aa3d071ccc0aff
allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84"
while read m r; do n=$(basename $m); /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) start $n"
  $PY s1k_rating.py --model $m --revision $r --stim ../stim --res1g $G --out $O > ../../logs/s1k_rating_$n.log 2>&1; rc=$?; echo "$(date -u +%FT%T) exit=$rc $n"
done <<< "$MODELS"
$PY s1k_pooled_tests.py --res $O > ../../logs/s1k_pooled_tests.log 2>&1; rc=$?; echo "$(date -u +%FT%T) exit=$rc s1k_pooled_tests.py"
echo QUEUE_DONE
