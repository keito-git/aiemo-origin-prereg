"""Stage 1q probe (no steering; before any Stage 1q registration). Each closing sentence of the twelve new magnitude
contrasts (stimuli/build_stage1q_closings.py) is rated alone, exactly as in s1p_probe.py, on extremity (Q_EXT),
emotion (Q_EMO), strength of feeling (Q_INT), pleasantness (Q_VAL) and arousal (Q_ARO). Reported per contrast: hi - lo
mean, frac_pos, hi and lo means. Used to check that extremity is matched between the pleasant and unpleasant groups.
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
from s1p_probe import Q_ARO

PLEASANT = ("brtK", "swtL", "sftM", "frgN", "qutP", "clrQ")
UNPLEASANT = ("nsyR", "stkS", "grsT", "btrU", "cldV", "drtW")
PAIRS = tuple((c, f"{c}_hi", f"{c}_lo") for c in PLEASANT + UNPLEASANT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cl = json.load(open(Path(a.stim) / "stage1q_closings.json"))
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
    (out / f"s1q_probe_{tag}.json").write_text(json.dumps({"args": vars(a), "res": res,
                                                           "t_total_s": time.time() - t0}, indent=1))
    print(tag)
    for name, r in res.items():
        print(f"  {name:6s} " + " ".join(f"{q} {r[q]['mean']:+.2f}" for q in ("ext", "emo", "int", "val", "aro")))


if __name__ == "__main__":
    main()
