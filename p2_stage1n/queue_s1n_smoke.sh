#!/bin/bash
# Stage 1n smoke (unplanned model Qwen3-8B-Base, tiny nulls) and code + readout-validity check. Run before the
# Stage 1n registration; it reports no steered outcome of a planned model.
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /root/aiemo/p2/code; PY=../../venv/bin/python
sha256sum s1n_rating.py s1n_pooled_tests.py s1m_rating.py s1l_rating.py s1j_rating.py ../stim/readout_R9n.csv
SM=/root/aiemo/results/smoke_s1n
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) smoke start"
$PY s1n_rating.py --model Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stim ../stim --res1m /root/aiemo/results/smoke_s1m --out $SM --n_pair 3 --n_null_minor 2 --n_pos 2 > ../../logs/s1n_smoke.log 2>&1; echo "$(date -u +%FT%T) smoke rc=$?"
$PY - <<"PYEOF" || { echo "SMOKE_CHECK_FAILED"; exit 1; }
import json
j = json.load(open("/root/aiemo/results/smoke_s1n/s1n_Qwen3-8B-Base_anger_d0.33.json"))
assert "Traceback" not in open("/root/aiemo/logs/s1n_smoke.log").read()
assert "t_total_s" in j and all(v > 0.99 for v in j["cos_with_1m"].values()), j["cos_with_1m"]
print("manipulation_max_abs_diff_1m", j["manipulation_max_abs_diff_1m"])
nl = j["readout"]["R9n"]["null"]
assert all(abs(v["var_check"] - 1) < 0.01 for k, v in nl.items() if "var_check" in v)
assert abs(nl["pair"]["var_check_neg"] - 1) < 0.01 and abs(nl["pair"]["var_check_neu"] - 1) < 0.01
assert nl["pair"]["cos_got_max_abs_err"] <= 0.01, nl["pair"]
c = j["checks"]
print({k: v for k, v in c.items() if k != "base_E"})
t = c["tasks"]
assert t["S"]["severity_effect"] > 1 and t["S"]["frac_pos"] >= 0.8, "S readout invalid"
assert t["C"]["rho_location"] > 0.5 and abs(t["C"]["severity_effect"]) < 1, "C readout invalid"
assert t["L"]["severity_effect"] < 0 and t["L"]["rho_with_S"] < 0, "L readout invalid"
assert c["repro_1m_SCL_first350_max_abs_diff"] < 0.02, "S/C/L baseline does not reproduce Stage 1m"
print("SMOKE_CHECK_OK")
PYEOF
echo SMOKE_DONE
