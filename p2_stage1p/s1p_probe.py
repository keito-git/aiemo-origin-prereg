"""Stage 1p probe (no steering; before any Stage 1p registration). Each closing sentence of every candidate intensity
contrast is rated alone, as in s1o_neu_probe.py, on extremity (Q_EXT), emotion (Q_EMO), strength of feeling (Q_INT),
pleasantness (Q_VAL) and arousal (Q_ARO, new). Reported per contrast: hi - lo mean, frac_pos, hi and lo means. Used to
choose the Stage 1p directions so that the extremity and emotion differences are not collinear across directions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1j_rating import Rating
from s1m_rating import Q_EMO, Q_EXT, Q_INT, Q_VAL
from s1o_cat_probe2 import Steer0

Q_ARO = ("On a scale from 1 (very calm) to 9 (very agitated or excited), how calm or agitated is the state this "
         "sentence describes? Answer with one digit.")
PAIRS = (("anger", "anger_L", "anger_lo"), ("joy", "joy_L", "joy_lo"), ("fear", "fear_L", "fear_lo"),
         ("sad", "sad_L", "sad_lo"), ("disg", "disg_L", "disg_lo"), ("grat", "grat_L", "grat_lo"),
         ("neu", "neu_hi", "neu_lo"), ("numA", "numA_hi", "numA_lo"), ("meaB", "meaB_hi", "meaB_lo"),
         ("plnC", "plnC_hi", "plnC_lo"), ("tmpD", "tmpD_hi", "tmpD_lo"), ("dstE", "dstE_hi", "dstE_lo"),
         ("wgtF", "wgtF_hi", "wgtF_lo"), ("slfG", "slfG_hi", "slfG_lo"),
         ("sizH", "sizH_hi", "sizH_lo"), ("qtyJ", "qtyJ_hi", "qtyJ_lo"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.stim)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cl = {}
    for f in ("stage1d", "stage1f", "stage1l", "stage1m", "stage1o_neu", "stage1p"):
        cl |= json.load(open(S / f"{f}_closings.json"))
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()

    def rate(q, key):
        it = [{"item_id": f"{key}_{i}", "prompt": f"Sentence: {s}\nQuestion: {q}\nAnswer:"}
              for i, s in enumerate(cl[key])]
        return Rating(tok, it).score(model, Steer0(), None)

    res = {}
    for name, hi, lo in PAIRS:
        r = {}
        for qn, q in (("ext", Q_EXT), ("emo", Q_EMO), ("int", Q_INT), ("val", Q_VAL), ("aro", Q_ARO)):
            eh, el = rate(q, hi), rate(q, lo)
            d = eh - el
            r[qn] = {"mean": float(d.mean()), "frac_pos": float((d > 0).mean()), "hi_mean": float(eh.mean()),
                     "lo_mean": float(el.mean())}
        res[name] = r
    tag = a.model.split("/")[-1]
    (out / f"s1p_probe_{tag}.json").write_text(json.dumps({"args": vars(a), "res": res,
                                                           "t_total_s": time.time() - t0}, indent=1))
    print(tag)
    for name, r in res.items():
        print(f"  {name:6s} " + " ".join(f"{q} {r[q]['mean']:+.2f}" for q in ("ext", "emo", "int", "val", "aro")))


if __name__ == "__main__":
    main()
