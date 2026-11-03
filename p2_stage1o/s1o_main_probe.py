"""Stage 1o probe 4 (no steering; before any Stage 1o registration): checks of the new 60 scenarios (L01..L60,
stimuli/build_r9o_main.py) on every readout, as the Stage 1k / 1n / 1o probes did on K01..K60.
Per readout, one score per scenario x severity:
  digit (S, HARSH, L, U_bright, U_fam, C): expected digit;
  4-choice (CAT_S, CAT_U, CAT_R, CAT_S2): expected semantic rank, mean over the four rotations (+ rotation consistency);
  PAIR: log-odds of the heavier option, mean over both orders and the three adjacent pairs (+ order consistency).
Summaries: severity effect (severe - mild), frac_pos, rho with S, rho_mild_severe; C also rho with the designed
location. The same summaries on the K scenarios are taken from earlier probes for comparison (not rerun here).
Writes s1o_mainprobe_{tag}.json."""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_light_round import load_csv
from s1o_cat_probe3 import LetterRating
from s1o_cat_probe2 import Steer0
from s1j_rating import Rating

DIGIT = ("S", "HARSH", "L", "U_bright", "U_fam", "C")
CAT4 = ("CAT_S", "CAT_U", "CAT_R", "CAT_S2")


def summ(x, xS, extra=None):
    n = len(x) // 2
    d = x[n:] - x[:n]
    return {"severity_effect": float(d.mean()), "frac_pos": float((d > 0).mean()), "frac_neg": float((d < 0).mean()),
            "rho_mild_severe": float(spearmanr(x[:n], x[n:])[0]), "rho_with_S": float(spearmanr(x, xS)[0]),
            "mean_mild": float(x[:n].mean()), "mean_severe": float(x[n:].mean()), **(extra or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    t0 = time.time()
    R = load_csv(Path(a.stim) / "readout_R9o_main.csv")
    for x in R:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    blk = lambda t, rot="": [x for x in R if x["task"] == t and x["rot"] == rot]
    ref = [(x["scen"], x["sev"]) for x in blk("S")]
    assert [s for _, s in ref] == ["mild"] * 60 + ["severe"] * 60
    sc = {}
    for t in DIGIT:
        it = blk(t)
        assert [(x["scen"], x["sev"]) for x in it] == ref, t
        sc[t] = Rating(tok, it).score(model, Steer0(), None)
    res = {}
    loc = np.array([int(x["loc"]) for x in blk("S")])
    for t in DIGIT:
        ex = {"rho_location": float(spearmanr(sc[t], loc)[0])} if t == "C" else None
        res[t] = summ(sc[t], sc["S"], ex)
    for t in CAT4:
        E, masses = [], []
        for rot in range(4):
            it = blk(t, str(rot))
            assert [(x["scen"], x["sev"]) for x in it] == ref, t
            P, mass = LetterRating(tok, it, "ABCD").probs(model)
            E.append((P * np.array([[int(c) for c in x["ranks"]] for x in it])).sum(1))
            masses.append(mass)
        E = np.stack(E)
        res[t] = summ(E.mean(0), sc["S"], {
            "rot_consistency": float(np.mean([spearmanr(E[i], E[j])[0] for i, j in combinations(range(4), 2)])),
            "letter_mass_mean": float(np.mean(masses))})
    LO, cons, by_pair, masses = [], [], {}, []
    for lo in range(3):
        lo_or = []
        for flip in range(2):
            it = blk("PAIR", f"{lo + 1}{lo + 2}{'ab'[flip]}")
            assert [(x["scen"], x["sev"]) for x in it] == ref
            P, mass = LetterRating(tok, it, "AB").probs(model)
            ph = np.clip(P[:, 1 if flip == 0 else 0], 1e-6, 1 - 1e-6)   # 'a' order: heavier at B
            lo_or.append(np.log(ph / (1 - ph)))
            masses.append(mass)
        lo_or = np.stack(lo_or)
        cons.append(float(spearmanr(lo_or[0], lo_or[1])[0]))
        LO.append(lo_or.mean(0))
        by_pair[f"{lo + 1}{lo + 2}"] = {"severity_effect": float((LO[-1][60:] - LO[-1][:60]).mean()),
                                        "order_consistency": cons[-1]}
    res["PAIR"] = summ(np.mean(LO, 0), sc["S"], {"order_consistency": float(np.mean(cons)), "by_pair": by_pair,
                                                 "letter_mass_mean": float(np.mean(masses))})
    log = {"args": vars(a), "res": res, "t_total_s": time.time() - t0}
    (out / f"s1o_mainprobe_{tag}.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({t: {k: round(v, 3) for k, v in r.items() if isinstance(v, float)} for t, r in res.items()}))


if __name__ == "__main__":
    main()
