"""Stage 1q exploratory analyses (NOT registered; run after s1q_pooled_tests.py).
Reads s1q_pooled_tests.json and the per-model files of the main run. All p-values here are exploratory.
  E1. H-val2 in the upper tail (the registered tail was lower; see the report for the sign issue).
  E2. H-val2 after regressing zJ_B on ext AND the projection z (is the valence gap carried by alignment?).
  E3. Unpleasant - pleasant gap of the projection z itself.
  E4. H-align after partialling ext, emo, val out of both zJ_B and the projection z.
  E5. Leave one model out H-align.
  E6. Within-model r of zJ_B with each human-style rating vs with the projection z; adjusted R2 of ratings-only vs
      ratings + projection z.
  E7. Discriminant check: r of the projection z with zC_B (control readouts).
  E8. Alternative predictors: projection on the raw (not control-residualized) mean judgment gradient, z-scored
      against the same nulls.
Permutations: 10000 draws, seed 9160.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1o_rating import COLS, JUDG
from s1q_rating import CONTRASTS, UNPLEASANT

SEED = 9160


def zstd(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    return y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def adj_r2(y, X):
    e = resid(y, X)
    n, k = len(y), X.shape[1]
    r2 = 1 - e.var() / y.var()
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--n_perm", type=int, default=10000)
    a = ap.parse_args()
    R = Path(a.res)
    T = json.load(open(R / "s1q_pooled_tests.json"))
    ms = T["models"]
    rng = np.random.default_rng(SEED)
    is_un = np.array([c in UNPLEASANT for c in CONTRASTS])
    per = {}
    for m in ms:
        d = T["descriptive"][m]
        zJ = np.array([d["zJ_B"][c] for c in CONTRASTS])
        zC = np.array([d["zC_B"][c] for c in CONTRASTS])
        P = np.array([d["proj_z"][c] for c in CONTRASTS])
        rat = {p: zstd([d["ratings"][c][p] for c in CONTRASTS]) for p in ("ext", "emo", "val")}
        # E8: raw mean judgment gradient (sign-adjusted, no control residualization), z against the same nulls
        pred = json.load(open(R / f"s1q_pred_{m}_d{a.depth}.json"))
        cols = pred["proj_cols"]
        ks = [cols.index(f"pred_{t}") for t in JUDG]
        sg = np.array([-1.0 if t == "L" else 1.0 for t in JUDG])
        raw = {}
        for c in CONTRASTS:
            pn = np.load(R / f"s1q_proj_null_{c}_{m}_d{a.depth}.npy")[:, ks] @ sg / len(JUDG)
            pv = np.array([pred["proj_dirs"][c][f"pred_{t}"] for t in JUDG]) @ sg / len(JUDG)
            raw[c] = float((pv - pn.mean()) / pn.std(ddof=1))
        per[m] = {"zJ": zJ, "zC": zC, "P": P, "rat": rat, "raw": np.array([raw[c] for c in CONTRASTS])}

    out = {"models": ms, "seed": SEED, "n_perm": a.n_perm, "note": "exploratory, not registered"}

    # E1/E2
    def gap_stat(ys, labels):
        return float(sum(y[l].mean() - y[~l].mean() for y, l in zip(ys, labels)))

    for key, Xf in (("E1_val2_upper", lambda p: p["rat"]["ext"][:, None]),
                    ("E2_val2_given_ext_proj", lambda p: np.column_stack([p["rat"]["ext"], p["P"]]))):
        ys = [resid(per[m]["zJ"], Xf(per[m])) for m in ms]
        obs = gap_stat(ys, [is_un] * len(ms))
        nul = np.array([gap_stat(ys, [rng.permutation(is_un) for _ in ms]) for _ in range(a.n_perm)])
        out[key] = {"obs": obs, "per_model": {m: float(y[is_un].mean() - y[~is_un].mean()) for m, y in zip(ms, ys)},
                    "p_upper": float((1 + np.sum(nul >= obs)) / (1 + a.n_perm)),
                    "p_two": float((1 + np.sum(np.abs(nul) >= abs(obs))) / (1 + a.n_perm)),
                    "n_models_positive": int(sum(y[is_un].mean() > y[~is_un].mean() for y in ys))}
    # E3
    out["E3_proj_gap"] = {m: float(per[m]["P"][is_un].mean() - per[m]["P"][~is_un].mean()) for m in ms}

    # E4/E5
    def align(Ps, Ys):
        return float(np.mean([np.arctanh(r(p, y)) for p, y in zip(Ps, Ys)]))

    def align_test(sel, partial):
        Ps, Ys = [], []
        for m in sel:
            p = per[m]
            X = np.column_stack([p["rat"][k] for k in ("ext", "emo", "val")])
            Ps.append(resid(p["P"], X) if partial else p["P"])
            Ys.append(resid(p["zJ"], X) if partial else p["zJ"])
        obs = align(Ps, Ys)
        nul = np.array([align([rng.permutation(x) for x in Ps], Ys) for _ in range(a.n_perm)])
        return {"obs": obs, "mean_r": float(np.tanh(obs)), "r_per_model": {m: r(x, y) for m, x, y in zip(sel, Ps, Ys)},
                "p_upper": float((1 + np.sum(nul >= obs)) / (1 + a.n_perm))}

    out["E4_align_partial_ratings"] = align_test(ms, True)
    out["E5_align_leave_one_model_out"] = {m: align_test([x for x in ms if x != m], False) for m in ms}

    # E6/E7/E8
    out["E6_r_with_zJ"] = {m: {**{k: r(per[m]["rat"][k], per[m]["zJ"]) for k in ("ext", "emo", "val")},
                               "proj": r(per[m]["P"], per[m]["zJ"]), "raw_grad": r(per[m]["raw"], per[m]["zJ"])}
                           for m in ms}
    out["E6_adj_r2"] = {m: {"ratings": adj_r2(per[m]["zJ"], np.column_stack([per[m]["rat"][k] for k in ("ext", "emo", "val")])),
                            "ratings_plus_proj": adj_r2(per[m]["zJ"], np.column_stack(
                                [per[m]["rat"][k] for k in ("ext", "emo", "val")] + [per[m]["P"]])),
                            "proj": adj_r2(per[m]["zJ"], per[m]["P"][:, None])} for m in ms}
    out["E7_r_proj_zC"] = {m: r(per[m]["P"], per[m]["zC"]) for m in ms}
    out["E8_r_proj_raw"] = {m: r(per[m]["P"], per[m]["raw"]) for m in ms}
    (R / "s1q_explore.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
