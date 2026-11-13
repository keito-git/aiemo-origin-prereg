"""Stage 1s: excess beyond alignment for the post-training checkpoints (design 02_設計/15_段階1s_..._draft_v2).
Per model, with the definitions of Stage 1p (zJ, s1p_pooled_tests.py) and Stage 1r (excess, s1r_excess.py) unchanged:
  zJ_c = mean over the valid judgment readouts of the null-z effect of contrast c (L sign reversed);
  excess_c = (zJ_c - null OLS line at the projection z of c) / SD of the null residuals.
Valid readouts come from s1s_validity.py (Stage 1o criteria). Tests per model over the 16 contrasts:
  one-sided t (mean excess > 0, df n - 1) and TOST (bounds +-BOUND); Holm across models is applied by the caller.
Diagnostics: slope and residual SD of the null line per contrast; number of valid judgment readouts.
CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from s1o_rating import COLS, JUDG
from s1p_rating import CONTRASTS

BOUND = 0.5
CATEGORY = ("anger", "joy", "fear", "sad", "disg", "grat", "neu")  # as s1r_excess.py (7 category, 9 magnitude)


def zc(x):
    return (x - x.mean()) / x.std(ddof=1)


def model_excess(R: Path, m: str, vj: list, depth: str) -> dict:
    j = json.load(open(R / f"s1p_{m}_d{depth}.json"))
    cols, pdirs = j["grad"]["proj_cols"], j["grad"]["proj_dirs"]
    k0 = cols.index("shared")
    sg = {t: (-1.0 if t == "L" else 1.0) for t in JUDG}
    ex, zJ, slope, rsd = {}, {}, {}, {}
    for c in CONTRASTS:
        nul = np.load(R / f"s1p_null_{c}_{m}_d{depth}.npy")
        # zJ exactly as s1p_pooled_tests.py
        zJ[c] = float(np.mean([sg[t] * (j["res"]["effects"][c][t] - nul[:, COLS.index(t)].mean())
                               / nul[:, COLS.index(t)].std(ddof=1) for t in vj]))
        # excess exactly as s1r_excess.py
        pn = np.load(R / f"s1p_proj_null_{c}_{m}_d{depth}.npy")[:, k0]
        y = np.mean([sg[t] * zc(nul[:, COLS.index(t)]) for t in vj], 0)
        p = zc(pn)
        b = np.polyfit(p, y, 1)
        pz = (pdirs[c]["shared"] - pn.mean()) / pn.std(ddof=1)
        r = np.std(y - np.polyval(b, p), ddof=2)
        ex[c] = float((zJ[c] - np.polyval(b, pz)) / r)
        slope[c], rsd[c] = float(b[0]), float(r)
    x = np.array([ex[c] for c in CONTRASTS])
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = x.mean() / se
    p_lo = stats.t.sf((x.mean() + BOUND) / se, n - 1)
    p_hi = stats.t.cdf((x.mean() - BOUND) / se, n - 1)
    is_emo = np.array([c in CATEGORY for c in CONTRASTS])
    return {"valid_judg": vj, "n_null": int(len(np.load(R / f"s1p_null_{CONTRASTS[0]}_{m}_d{depth}.npy"))),
            "excess": ex, "zJ": zJ, "null_slope": slope, "null_resid_sd": rsd,
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n_pos": int((x > 0).sum()), "n": n,
            "t": float(t), "p_one_sided": float(stats.t.sf(t, n - 1)),
            "tost": {"p_lower": float(p_lo), "p_upper": float(p_hi), "p": float(max(p_lo, p_hi))},
            "mean_category": float(x[is_emo].mean()), "mean_magnitude": float(x[~is_emo].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="directory with s1p_rating.py outputs")
    ap.add_argument("--valid", required=True, help="s1s_validity.py output")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V = json.load(open(a.valid))
    res = {}
    for m in a.models:
        if not V[m].get("enter"):
            res[m] = {"status": "not measurable (readout validity)"}
            continue
        res[m] = model_excess(Path(a.res), m, V[m]["valid_judgment"], a.depth)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        if "mean" not in r:
            print(m, r["status"])
            continue
        print(f"{m}: n_null {r['n_null']} mean {r['mean']:+.2f} sd {r['sd']:.2f} pos {r['n_pos']}/{r['n']} "
              f"t {r['t']:.2f} p1 {r['p_one_sided']:.4f} TOST p {r['tost']['p']:.3f} "
              f"cat {r['mean_category']:+.2f} mag {r['mean_magnitude']:+.2f} "
              f"slope med {np.median(list(r['null_slope'].values())):.2f} "
              f"rsd med {np.median(list(r['null_resid_sd'].values())):.2f}")


if __name__ == "__main__":
    main()
