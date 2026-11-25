#!/bin/bash
# P3-K direction set at the chosen layer (prereg v3 §3). No task gradient is computed here.
# Waits for run_p3k_prep.sh (P3K_PREP_DONE in its log), then builds and assembles:
#   contrast (contrasts_v1.json, discovery pairs): app (A), ctrl, mag, rand_nat
#   cat (GoEmotions single-label, <= 400 rows per label): cat (A), H_cat2/H_cat3
#   vad-ext EmoBank (sentences, V/A): va_ext (A), H_VA_ext_emobank
#   vad-ext NRC-VAD v2.1 unigrams (V/A): va_ext (A), H_VA_ext_nrc
#   sun-va at the chosen layer (from run_p3k_prep.sh): va_sun (A), H_VA_sun
#   sun-matched at the chosen layer (v3.1): H_VA_sun_matched (K1 candidate only, not in A)
# Launch (detached):  setsid nohup bash build_dirs_p3k.sh > /root/aiemo/logs/p3k_dirs.log 2>&1 < /dev/null &
set -u
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/aiemo/p3/code; PY=../../venv/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659
DATA=/root/aiemo/p3/data; O=/root/aiemo/results/p3_k/dirs
NRC=$DATA/nrc/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt
GSUB=$DATA/goemotions/goemotions_single_sub400.csv; EB=$DATA/emobank/emobank.csv

until grep -q -E "P3K_PREP_DONE|SUN_REPRO_FAILED|SUN_MATCHED_FAILED" /root/aiemo/logs/p3k_prep.log 2>/dev/null; do sleep 300; done
grep -q P3K_PREP_DONE /root/aiemo/logs/p3k_prep.log || { echo "prep did not finish (SUN_REPRO_FAILED?)"; exit 1; }
LAYER=$($PY -c "import json; print(json.load(open('$O/select_layer.json'))['chosen'])")
echo "layer $LAYER"; sha256sum p3k_directions.py contrasts_v1.json $EB $NRC $GSUB
( sha256sum -c --quiet contrasts_v1.sha256 ) || { echo "CONTRAST_HASH_FAILED"; exit 1; }

run() {  # name, then the p3k_directions.py arguments; retried on failure (shared GPU), skipped if the output exists
  local out=$1; shift
  [ -f "$out" ] && { echo "exists $out"; return 0; }
  for try in 1 2 3 4 5 6; do
    /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) $* (try $try)"
    $PY p3k_directions.py "$@" --out "$out" && return 0
    sleep 600
  done
  echo "FAILED $out"; exit 1
}
G="--model $MODEL --revision $REV --layer $LAYER"
run $O/contrast_v1_L$LAYER.npz contrast $G --spec contrasts_v1.json --splits disc
run $O/cat_L$LAYER.npz cat $G --data $GSUB
run $O/vad_emobank_L$LAYER.npz vad-ext $G --data $EB --format emobank --dims V,A --name emobank --save_pattern
run $O/vad_nrc_L$LAYER.npz vad-ext $G --data $NRC --format nrc --dims V,A --name nrc --save_pattern
$PY p3k_directions.py assemble --parts $O/sun_va_L$LAYER.npz $O/cat_L$LAYER.npz $O/vad_emobank_L$LAYER.npz \
  $O/vad_nrc_L$LAYER.npz $O/contrast_v1_L$LAYER.npz $O/sun_matched_L$LAYER.npz --out $O/directions_L$LAYER.npz || exit 1
sha256sum $O/directions_L$LAYER.npz
echo "$(date -u +%FT%T) P3K_DIRS_DONE layer=$LAYER"
