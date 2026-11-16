"""Stage 1t registered tests (CPU only; reads s1s_excess.py output for the base models and the Stage 1s output).
Design 02_設計/16_段階1t_..._draft_20261116.
H-base (primary): Llama-3.1-8B and OLMo-2-1124-7B (base); one-sided t of mean excess > 0 (df 15), Holm over the two.
  "positive" = Holm-adjusted p < .05; TOST (+-0.5) per base, "zero (TOST)" only if TOST p < .05.
H-shift (secondary, Llama family only): paired t over the 16 directions of (Tulu-3-8B-SFT - Llama-3.1-8B), two-sided.
H-data trigger: Llama-3.1-8B "not positive" (zero or undecided). Not run if positive or not measurable.
A base whose readouts fail the Stage 1o validity rule is "not measurable" and is dropped from the Holm family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from s1p_rating import CONTRASTS
from s1s_tests import family

BASE = ("Llama-3.1-8B", "OLMo-2-1124-7B")
SFT = "Llama-3.1-Tulu-3-8B-SFT"

READING = {
    "positive": "residual present before post-training (pretraining) and kept through post-training",
    "zero (TOST)": "residual arises in post-training (by SFT)",
    "not positive, undecided": "suggests the residual arises in post-training (base not shown zero)",
    "not measurable": "stage attribution stays 'present at the SFT checkpoint' (limitation)",
    "missing": "missing",
}


def shift(E: dict, S: dict) -> dict:
    if "mean" not in E.get(BASE[0], {}) or "mean" not in S.get(SFT, {}):
        return {"status": "not testable"}
    d = np.array([S[SFT]["excess"][c] - E[BASE[0]]["excess"][c] for c in CONTRASTS])
    t, p = stats.ttest_1samp(d, 0.0)
    return {"mean_diff_sft_minus_base": float(d.mean()), "sd": float(d.std(ddof=1)), "n_pos": int((d > 0).sum()),
            "t": float(t), "p_two_sided": float(p), "significant": bool(p < .05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excess", required=True, help="s1s_excess.py output for the two base models")
    ap.add_argument("--excess_s1s", required=True, help="Stage 1s s1s_excess.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E, S = json.load(open(a.excess)), json.load(open(a.excess_s1s))
    hb = family(E, BASE)
    hs = shift(E, S)
    rd = {m: READING[hb[m]["status"]] for m in BASE}
    if hb[BASE[0]]["status"] == "positive" and hs.get("status") != "not testable":
        rd[BASE[0]] += "; " + ("strengthened at SFT" if hs["significant"] and hs["mean_diff_sft_minus_base"] > 0
                              else "weakened at SFT" if hs["significant"] else "not clearly changed at SFT")
    trig = hb[BASE[0]]["status"] in ("zero (TOST)", "not positive, undecided")
    out = {"H-base": hb, "H-shift": hs, "reading": rd, "H-data_triggered": trig}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
