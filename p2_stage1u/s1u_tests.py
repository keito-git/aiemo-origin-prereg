"""Stage 1u registered tests (CPU only; reads the s1s_excess.py output of the five checkpoints run with the F8 prefix).
Design 02_設計/17_段階1u_..._draft_20261118.
H-format (read first): Tulu-3-8B-SFT, OLMo-2-1124-7B-SFT, Llama-3.1-8B-Instruct; one-sided t (df 15), Holm over three.
  The F8 format "measures the residual" only if at least two of the three are positive; in addition, a base is read
  (origin claim) only if the SFT checkpoint of its own family is positive with F8 (R2, 11/18).
H-base (primary): Llama-3.1-8B and OLMo-2-1124-7B; one-sided t of mean excess > 0, Holm over the two;
  TOST (+-0.5) per base, "zero (TOST)" only if TOST p < .05. Read only if H-format holds.
H-shift (secondary): paired t over the 16 directions of (SFT - base) per family, two-sided, Holm over the two families.
Format scale (descriptive, R2/R3 11/18): for the three post-trained checkpoints, F8 excess vs the Stage 1s / 1p excess
  without prefix (--excess_s1s): ratio of the means and Pearson r over the 16 directions; a ratio outside [0.5, 2]
  is flagged "scale changed by the format" and then no magnitude is compared across formats.
A checkpoint whose readouts fail the Stage 1o validity rule is "not measurable" and is dropped from its family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from s1p_rating import CONTRASTS
from s1s_tests import family, holm

POST = ("Llama-3.1-Tulu-3-8B-SFT", "OLMo-2-1124-7B-SFT", "Llama-3.1-8B-Instruct")
BASE = ("Llama-3.1-8B", "OLMo-2-1124-7B")
SFT_OF = {"Llama-3.1-8B": "Llama-3.1-Tulu-3-8B-SFT", "OLMo-2-1124-7B": "OLMo-2-1124-7B-SFT"}

READING = {
    "positive": "residual present before post-training (pretraining) and kept through post-training",
    "zero (TOST)": "residual arises in post-training (by SFT)",
    "not positive, undecided": "suggests the residual arises in post-training (base not shown zero)",
    "not measurable": "origin not determined (as Stage 1t)",
    "missing": "missing",
}


def shift(E: dict) -> dict:
    ps, res = {}, {}
    for b, s in SFT_OF.items():
        if "mean" not in E.get(b, {}) or "mean" not in E.get(s, {}):
            res[b] = {"status": "not testable"}
            continue
        d = np.array([E[s]["excess"][c] - E[b]["excess"][c] for c in CONTRASTS])
        t, p = stats.ttest_1samp(d, 0.0)
        res[b] = {"sft": s, "mean_diff_sft_minus_base": float(d.mean()), "sd": float(d.std(ddof=1)),
                  "n_pos": int((d > 0).sum()), "t": float(t), "p_two_sided": float(p)}
        ps[b] = float(p)
    for b, q in holm(ps).items():
        res[b]["p_holm"] = q
        res[b]["significant"] = bool(q < .05)
    return res


def scale(E: dict, S: dict) -> dict:
    res = {}
    for m in POST:
        if "mean" not in E.get(m, {}) or "mean" not in S.get(m, {}):
            res[m] = {"status": "not comparable"}
            continue
        f8 = np.array([E[m]["excess"][c] for c in CONTRASTS])
        f0 = np.array([S[m]["excess"][c] for c in CONTRASTS])
        ratio = float(f8.mean() / f0.mean()) if f0.mean() != 0 else float("nan")
        res[m] = {"mean_f8": float(f8.mean()), "mean_f0": float(f0.mean()), "ratio": ratio,
                  "r": float(stats.pearsonr(f8, f0)[0]), "flag_scale_changed": not (0.5 <= ratio <= 2.0)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excess", required=True, help="s1s_excess.py output for the five F8 checkpoints")
    ap.add_argument("--excess_s1s", nargs="+", default=[],
                    help="no-prefix excess files (Stage 1s s1s_excess.json; Stage 1r s1r_excess_s1p.json format)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = json.load(open(a.excess))
    S = {}
    for f in a.excess_s1s:
        S.update(json.load(open(f)))
    hf = family(E, POST)
    ok = sum(hf[m]["status"] == "positive" for m in POST) >= 2
    hb = family(E, BASE)
    hs = shift(E)
    rd = {}
    for b in BASE:
        if not ok:
            rd[b] = "F8 format not comparable (H-format failed): no origin claim"
            continue
        if hf.get(SFT_OF[b], {}).get("status") != "positive":
            rd[b] = (f"family check failed ({SFT_OF[b]} not positive with F8): base result reported, "
                     "no origin claim for this family")
            continue
        rd[b] = READING[hb[b]["status"]]
        h = hs[b]
        if hb[b]["status"] == "positive" and "significant" in h:
            rd[b] += "; " + ("strengthened at SFT" if h["significant"] and h["mean_diff_sft_minus_base"] > 0
                             else "weakened at SFT" if h["significant"] else "not clearly changed at SFT")
    out = {"H-format": hf, "H-format_holds": ok, "H-base": hb, "H-shift": hs, "reading": rd,
           "format_scale_descriptive": scale(E, S)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
