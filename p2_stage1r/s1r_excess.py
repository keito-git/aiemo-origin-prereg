"""Stage 1r: do emotion directions move judgments more than random directions of equal alignment?
Registered before running on the Stage 1p data (aiemo-origin-prereg P2-S1r_preregistration.md).
Per model and contrast c, with the n_null variance-matched random directions of c as the reference population:
  y_null = mean over the valid judgment readouts of the null effect, each readout z-scored over the nulls of c (L sign
  reversed); p_null = projection of each null on the shared judgment gradient, z-scored over the nulls of c.
  Null line: OLS of y_null on p_null. excess_c = (zJ_c - line(projz_c)) / SD of the null residuals.
  zJ_c and projz_c are the stage's own values (Stage 1p: zJ from s1p_pooled_tests.json, projection z on the shared
  gradient saved before steering).
Tests (all models except Llama-3.1-Tulu-3-8B enter H-excess; Tulu is tested separately):
  H-excess (primary, equivalence): m = mean over models of the per-model mean excess; SE = SD of the per-model means /
    sqrt(n_models); TOST with bounds +-BOUND, t with n_models - 1 df, alpha .05 each side.
  H-tulu (secondary, one-sided upper): mean excess of Tulu over contrasts > 0, t with n_contrasts - 1 df; sign test
    reported alongside.
Descriptive: per-model mean excess, n positive, the same with Tulu included, emotion-category vs magnitude contrasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from s1o_rating import COLS, JUDG

TULU = "Llama-3.1-Tulu-3-8B"
BOUND = 0.5
CATEGORY = ("anger", "joy", "fear", "sad", "disg", "grat", "neu")


def zc(x):
    return (x - x.mean()) / x.std(ddof=1)


def excess_for_model(R: Path, stage: str, m: str, T: dict, depth: str):
    j = json.load(open(R / f"{stage}_{m}_d{depth}.json"))
    if stage == "s1p":
        C, cols, pdirs = j["contrasts"], j["grad"]["proj_cols"], j["grad"]["proj_dirs"]
        zJ = T["descriptive"][m]["zJ"]
        vj = T["descriptive"][m]["valid_judg"]
    else:
        pred = json.load(open(R / f"{stage}_pred_{m}_d{depth}.json"))
        C, cols, pdirs = j["contrasts"], pred["proj_cols"], pred["proj_dirs"]
        zJ = T["descriptive"][m]["zJ_B"]
        vj = T["descriptive"][m]["valid_judg"]
    k0 = cols.index("shared")
    ex = {}
    for c in C:
        nul = np.load(R / f"{stage}_null_{c}_{m}_d{depth}.npy")
        pn = np.load(R / f"{stage}_proj_null_{c}_{m}_d{depth}.npy")[:, k0]
        y = np.mean([(-1.0 if t == "L" else 1.0) * zc(nul[:, COLS.index(t)]) for t in vj], 0)
        p = zc(pn)
        b = np.polyfit(p, y, 1)
        pz = (pdirs[c]["shared"] - pn.mean()) / pn.std(ddof=1)
        ex[c] = float((zJ[c] - np.polyval(b, pz)) / np.std(y - np.polyval(b, p), ddof=2))
    return ex


def tost(means):
    n = len(means)
    m, se = float(np.mean(means)), float(np.std(means, ddof=1) / np.sqrt(n))
    t_lo, t_hi = (m + BOUND) / se, (BOUND - m) / se
    p_lo, p_hi = float(stats.t.sf(t_lo, n - 1)), float(stats.t.sf(t_hi, n - 1))
    ci90 = [m - stats.t.ppf(.95, n - 1) * se, m + stats.t.ppf(.95, n - 1) * se]
    return {"mean": m, "se": se, "n_models": n, "t_lower": t_lo, "t_upper": t_hi, "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": max(p_lo, p_hi), "ci90": ci90, "bound": BOUND, "equivalent": bool(max(p_lo, p_hi) < .05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--stage", choices=("s1p", "s1q"), required=True)
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.res)
    T = json.load(open(R / f"{a.stage}_pooled_tests.json"))
    ex = {m: excess_for_model(R, a.stage, m, T, a.depth) for m in T["models"]}
    means = {m: float(np.mean(list(v.values()))) for m, v in ex.items()}
    out = {"stage": a.stage, "models": T["models"], "bound": BOUND, "excess": ex, "model_mean": means,
           "model_n_pos": {m: int(sum(x > 0 for x in v.values())) for m, v in ex.items()}}
    nt = [m for m in T["models"] if m != TULU]
    out["H-excess"] = tost([means[m] for m in nt])
    out["H-excess_with_tulu_descriptive"] = tost([means[m] for m in T["models"]])
    if TULU in ex:
        x = np.array(list(ex[TULU].values()))
        t = stats.ttest_1samp(x, 0.0, alternative="greater")
        out["H-tulu"] = {"mean": float(x.mean()), "t": float(t.statistic), "df": len(x) - 1, "p": float(t.pvalue),
                         "n_pos": int((x > 0).sum()), "n": len(x),
                         "p_sign": float(stats.binomtest(int((x > 0).sum()), len(x), .5, "greater").pvalue)}
    if a.stage == "s1p":
        out["by_type"] = {m: {"category": float(np.mean([v[c] for c in v if c in CATEGORY])),
                              "magnitude": float(np.mean([v[c] for c in v if c not in CATEGORY]))}
                          for m, v in ex.items()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k.startswith("H") or k in ("model_mean", "model_n_pos", "by_type")},
                     indent=1))


if __name__ == "__main__":
    main()
