#!/bin/bash
# P3-K kill experiment, gradient stage (prereg draft 02_設計/04_事前登録草案_撤退実験_v2). Mirrors the P2 queue scripts.
# Launch (detached):  LAYER=<L> setsid nohup bash queue_p3k.sh > /root/aiemo/logs/queue_p3k.log 2>&1 < /dev/null &
# LAYER must be the layer fixed by `p3k_directions.py select-layer` BEFORE any task gradient is computed (prereg §2).
# Gates, in order: stimulus hashes -> label tokens single (verify_label_tokens.py) -> smoke run (finite differences)
# -> full gradient run (resumable) -> kill tests only if DIRS (assembled directions file) is given.
set -u
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
cd /root/aiemo/p3/code; PY=../../venv/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659   # revision as in P2 Stage 1u
STIM=stimuli_v1   # verify_label_tokens.py reads code/stimuli_v1 (build_p3k_stimuli.OUT); maps 1-4 = stimuli_v0
O=/root/aiemo/results/p3_k; SM=/root/aiemo/results/smoke_p3k; LOG=../../logs
mkdir -p $O $SM
: "${LAYER:?set LAYER (chosen by p3k_directions.py select-layer)}"
sha256sum p3k_common.py p3k_gradients.py p3k_directions.py p3k_kill_tests.py verify_label_tokens.py

# gate 0: stimuli unchanged
( cd $STIM && sha256sum -c --quiet SHA256SUMS ) || { echo "STIM_HASH_FAILED"; exit 1; }

# gate 1: every label is one token in the reference tokenizer (stimuli README, open item 1)
TJ=$(ls $HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/$REV/tokenizer.json)
$PY verify_label_tokens.py $TJ $O/label_token_check_server.json > $LOG/p3k_labels.log 2>&1
$PY - <<PYEOF || { echo "LABEL_CHECK_FAILED"; exit 1; }
import json
j = json.load(open("$O/label_token_check_server.json"))
assert j["all_bare_single"] and j["all_space_single"] and j["n_in_context_failures"] == 0, j
print("LABEL_CHECK_OK")
PYEOF

# gate 2: smoke (V1, maps 1 and 5 (Cyrillic labels), 4 scenarios, 3 nulls, finite-difference check)
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) smoke start"
$PY p3k_gradients.py --model $MODEL --revision $REV --stim $STIM --out $SM --layer $LAYER --tasks V1 --maps 1,5 \
  --baselines C1,C3 --max_scen 4 --n_null 3 --n_splits 5 --fd_check 5 > $LOG/p3k_smoke.log 2>&1
echo "$(date -u +%FT%T) smoke rc=$?"
$PY - <<PYEOF || { echo "SMOKE_CHECK_FAILED"; exit 1; }
import glob, json
assert "Traceback" not in open("$LOG/p3k_smoke.log").read()
d = sorted(glob.glob("$SM/*_L$LAYER"))[-1]
r = json.load(open(sorted(glob.glob(d + "/run_*.json"))[-1]))
assert "t_total_s" in r
for f in ("shared_V1_map1.npz", "shared_V1_map5.npz", "act_stats.npz", "unembed.npz", "base_C1.npz", "base_C3.npz"):
    open(f"{d}/{f}", "rb").close()
fd = r["fd_check"]
print("fd corr_random", fd["corr_random"], "g_ratio_alpha16", fd["g_ratio_alpha16"])
# P2 Stage 1n amendment 01 criterion; no validity or effect value is read here
assert fd["corr_random"] >= 0.95 and 0.9 <= fd["g_ratio_alpha16"] <= 1.1, "gradient does not match finite differences"
print("SMOKE_CHECK_OK")
PYEOF

# full run: 28 tasks x 6 maps, C1-C4, 100 nulls per readout, 100 scenario splits (resumable per unit)
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) gradients start"
$PY p3k_gradients.py --model $MODEL --revision $REV --stim $STIM --out $O --layer $LAYER \
  --n_null 100 --n_splits 100 > $LOG/p3k_gradients.log 2>&1
echo "$(date -u +%FT%T) gradients rc=$?"

# kill tests (CPU) once the directions file exists (human-rating parts and H_VA^Sun are TODO)
GD=$O/$(basename $MODEL)_L$LAYER
if [ -n "${DIRS:-}" ] && [ -f "$DIRS" ]; then
  $PY p3k_kill_tests.py --grad_dir $GD --dirs $DIRS --out $O/p3k_verdict.json > $LOG/p3k_kill.log 2>&1
  echo "$(date -u +%FT%T) kill tests rc=$?"
else
  echo "DIRS not set: run p3k_kill_tests.py after the directions are assembled"
fi
echo QUEUE_P3K_DONE
