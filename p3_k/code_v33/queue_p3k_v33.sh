#!/bin/bash
# P3-K prereg v3.3, confirmatory cohort (V17-V40, N13-N36). Run only after the prereg v3.3 is public.
# Launch (detached): setsid nohup bash queue_p3k_v33.sh > /root/aiemo/logs/queue_p3k_v33.log 2>&1 < /dev/null &
# Order: stimulus hashes -> eligibility test (ONCE; J and CTRL on the pilot scenarios) -> smoke (finite differences)
# -> gradients for the eligible tasks (maps 1, 2, 3, 7, 8; CAT effect threshold 0.375; PAIR 13/24/14)
# -> kill tests (task entry in maps 1 and 2; coverage gate; K3 map gate on 3, 7, 8). Directions and layer from v3.1.
set -u
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
cd /root/aiemo/p3/code; PY=../../venv/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659
STIM=stimuli_v3; LAYER=13; MAPS=1,2,3,7,8
O=/root/aiemo/results/p3_k_v33; SM=/root/aiemo/results/smoke_p3k_v33; LOG=../../logs
DIRS=/root/aiemo/results/p3_k/dirs/directions_L13.npz
mkdir -p $O $SM
sha256sum p3k_common.py p3k_gradients.py p3k_kill_tests.py p3k_pilot.py build_p3k_stimuli_v3.py $DIRS

# gate 0: stimuli unchanged since the prereg
( cd $STIM && sha256sum -c --quiet SHA256SUMS ) || { echo "STIM_HASH_FAILED"; exit 1; }

# eligibility test (run once; p3k_pilot.py refuses to overwrite its output)
if [ ! -f $O/p3k_eligibility.json ]; then
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) eligibility start"
  $PY p3k_pilot.py --model $MODEL --revision $REV --pilot $STIM/p3k_pilot.csv --out $O > $LOG/p3k_v33_pilot.log 2>&1
  echo "$(date -u +%FT%T) eligibility rc=$?"
fi
ELIG=$($PY -c "import json; print(','.join(json.load(open('$O/p3k_eligibility.json'))['eligible']))")
echo "eligible: $ELIG"
[ -n "$ELIG" ] || { echo "NO_ELIGIBLE_TASKS"; exit 1; }
FIRST=${ELIG%%,*}

# smoke: first eligible task, maps 1 and 7, 4 scenarios, finite-difference check
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) smoke start"
$PY p3k_gradients.py --model $MODEL --revision $REV --stim $STIM --out $SM --layer $LAYER --tasks $FIRST --maps 1,7 \
  --baselines "" --max_scen 4 --n_null 3 --n_splits 5 --fd_check 5 --cat_effect_min 0.375 > $LOG/p3k_v33_smoke.log 2>&1
echo "$(date -u +%FT%T) smoke rc=$?"
$PY - <<PYEOF || { echo "SMOKE_CHECK_FAILED"; exit 1; }
import glob, json
assert "Traceback" not in open("$LOG/p3k_v33_smoke.log").read()
d = sorted(glob.glob("$SM/*_L$LAYER"))[-1]
r = json.load(open(sorted(glob.glob(d + "/run_*.json"))[-1]))
fd = r["fd_check"]
print("fd corr_random", fd["corr_random"], "g_ratio_alpha16", fd["g_ratio_alpha16"])
assert fd["corr_random"] >= 0.95 and 0.9 <= fd["g_ratio_alpha16"] <= 1.1, "gradient does not match finite differences"
print("SMOKE_CHECK_OK")
PYEOF

# gradients (resumable per unit)
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) gradients start"
$PY p3k_gradients.py --model $MODEL --revision $REV --stim $STIM --out $O --layer $LAYER --tasks $ELIG --maps $MAPS \
  --baselines "" --n_null 100 --n_splits 100 --cat_effect_min 0.375 > $LOG/p3k_v33_gradients.log 2>&1
echo "$(date -u +%FT%T) gradients rc=$?"

# kill tests (CPU)
GD=$O/$(basename $MODEL)_L$LAYER
$PY p3k_kill_tests.py --grad_dir $GD --dirs $DIRS --out $O/p3k_verdict_v33.json --plan p3k_v3_task_plan.json \
  --eligible $O/p3k_eligibility.json --validity_maps 1,2 --main_maps 1,2 --k3_maps 3,7,8 > $LOG/p3k_v33_kill.log 2>&1
echo "$(date -u +%FT%T) kill tests rc=$?"
echo QUEUE_P3K_V33_DONE
