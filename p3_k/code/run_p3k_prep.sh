#!/bin/bash
# P3-K preparation before any task gradient (prereg v1 §2; gate revised in v3.1 §10): Sun VA reproduction over all layers, then layer selection.
# Launch (detached):  setsid nohup bash run_p3k_prep.sh > /root/aiemo/logs/p3k_prep.log 2>&1 < /dev/null &
# Layer rule (fixed here, before the run):
#   1. sun-va with self-report targets at hidden_states 1..32; L* = argmax over layers of mean(r_V, r_A) (in-sample,
#      the value comparable to Sun's report).
#   2. candidates = {L*-4, L*-2, L*, L*+2, L*+4} within [2, 30]; select-layer (category-direction split-half
#      reproducibility on GoEmotions single-label, <= 400 rows per label, seed 20261124) picks the layer.
#   3. sun-va refit on the CPU at the chosen layer (cached vectors and ratings).
set -u
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
cd /root/aiemo/p3/code; PY=../../venv/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659
DATA=/root/aiemo/p3/data; O=/root/aiemo/results/p3_k/dirs; mkdir -p $O
NRC=$DATA/nrc/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt
GE=$DATA/goemotions/goemotions_raw_all.csv; GSUB=$DATA/goemotions/goemotions_single_sub400.csv
ALL=$(seq -s, 1 32)
sha256sum p3k_directions.py p3k_common.py p3k_gradients.py $GE $GSUB $NRC

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# retry on failure (e.g. OOM when another project starts a job on the shared GPU); vectors/ratings are cached once done
for try in 1 2 3 4 5 6; do
  /root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) sun-va (all layers) start, try $try"
  $PY p3k_directions.py sun-va --model $MODEL --revision $REV --layer 16 --check_layers $ALL --data $GE --bs 8 \
    --ratings $O/sun_self_report.json --nrc $NRC --out $O/sun_va_L16.npz --out_nrc $O/sun_va_nrc_L16.npz && break
  [ $try = 6 ] && exit 1; sleep 600
done
echo "$(date -u +%FT%T) sun-va rc=0"

# reproduction gate (prereg v3.1 §10). The registered H_VA^Sun (LOLO-chosen k, lambda) must reproduce valence
# (in-sample r_V >= 0.90 at some layer). Arousal is not gated on it: on 27 labels the in-sample r_A depends on k
# (v3 gate failed: max r_A = .65 with LOLO k), so Sun's .87 is matched by H_VA^Sun-matched (p3k_sun_matched.py),
# which is required below and used only in the K1 sensitivity analysis (prereg v3.1 §8; external review 3).
$PY - <<PYEOF || { echo "SUN_REPRO_FAILED"; exit 1; }
import json
r = json.load(open("$O/sun_va_L16.report.json"))["fits"]["self"]["per_layer"]
bv, ba = max(v["V"]["r_insample"] for v in r.values()), max(v["A"]["r_insample"] for v in r.values())
print("registered fit: max r_V", round(bv, 3), "max r_A", round(ba, 3))
assert bv >= 0.90, "Sun valence not reproduced"
print("SUN_V_REPRO_OK")
PYEOF
CAND=$($PY - <<PYEOF
import json
r = json.load(open("$O/sun_va_L16.report.json"))["fits"]["self"]["per_layer"]
sc = {int(L): (v["V"]["r_insample"] + v["A"]["r_insample"]) / 2 for L, v in r.items()}
Ls = max(sc, key=sc.get)
print(",".join(str(L) for L in (Ls - 4, Ls - 2, Ls, Ls + 2, Ls + 4) if 2 <= L <= 30))
PYEOF
)
echo "candidate layers: $CAND"
/root/aiemo/code/wait_gpu_free.sh 30000; echo "$(date -u +%FT%T) select-layer start"
$PY p3k_directions.py select-layer --model $MODEL --revision $REV --data $GSUB --layers $CAND \
  --out $O/select_layer.json || exit 1
LAYER=$($PY -c "import json; print(json.load(open('$O/select_layer.json'))['chosen'])")
echo "chosen layer: $LAYER"
# reuse the all-layer steering vectors (same data, layers and code); only the word check needs the GPU
[ -f $O/sun_va_L$LAYER.vectors.npz ] || cp $O/sun_va_L16.vectors.npz $O/sun_va_L$LAYER.vectors.npz
/root/aiemo/code/wait_gpu_free.sh 30000
$PY p3k_directions.py sun-va --layer $LAYER --check_layers $ALL --data $GE --ratings $O/sun_self_report.json \
  --nrc $NRC --word_check 2000 --model $MODEL --revision $REV \
  --out $O/sun_va_L$LAYER.npz --out_nrc $O/sun_va_nrc_L$LAYER.npz || exit 1
$PY p3k_sun_matched.py --vectors $O/sun_va_L16.vectors.npz --ratings $O/sun_self_report.json --layer $LAYER \
  --out $O/sun_matched_L$LAYER.npz || { echo "SUN_MATCHED_FAILED"; exit 1; }
echo "$(date -u +%FT%T) P3K_PREP_DONE layer=$LAYER"
