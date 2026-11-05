"""Stage 1p pooled tests (design 02_設計/12_段階1p_感情か大きさか_設計_draft_20261105.md; registered before any
Stage 1p steered run of a planned model).
Per model and contrast c: z of every readout against the n_null variance-matched nulls of c (L sign-reversed);
zJ = mean z over the valid judgment readouts, zC = mean z over the valid controls. Readout validity is taken from the
Stage 1o pooled tests (same models, items and readouts; no steering).
Predictors per model: the hi - lo ratings of the closing sentences measured in the same run (ext, emo, val),
standardized across the contrasts of that model.
  H-mag (primary, one-sided): sum over models of the OLS coefficient of ext in zJ ~ 1 + ext + emo + val.
  H-emo (two-sided): the same for emo.  H-val (two-sided): the same for val.
  Inference: Freedman-Lane permutation (residuals of the reduced model permuted within model; the same permutation
  index is not shared across models), n_perm draws, seed 8950; p = (1 + #{null >= obs}) / (1 + n_perm).
  H-cat (two-sided): mean over models of [mean zJ of the emotion contrasts - mean zJ of the magnitude contrasts] after
  regressing zJ on ext within model; permutation of the category labels within model.
  H-sel (two-sided): as H-cat with zJ - zC.
Descriptive: per model coefficients; zJ, zC and the ratings per contrast; L breakdown (z_L - mean z of the other valid
judgment readouts) against the digit-control z (mean of U_bright, U_fam).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1o_rating import COLS, CTRL, JUDG
from s1p_rating import CONTRASTS, EMOTION

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
PRED = ("ext", "emo", "val")
SEED = 8950


def zstd(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    return b[1:], X1 @ b


def fl_coef_null(y, X, k, rng, n_perm):
    """Freedman-Lane null of the coefficient of column k: permute the residuals of y ~ X without column k."""
    Xr = np.delete(X, k, 1)
    _, fit_r = ols(y, Xr)
    e = y - fit_r
    out = np.empty(n_perm)
    for i in range(n_perm):
        out[i] = ols(fit_r + rng.permutation(e), X)[0][k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--valid_from", required=True, help="Stage 1o s1o_pooled_tests.json")
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--n_perm", type=int, default=10000)
    a = ap.parse_args()
    R = Path(a.res)
    V1o = json.load(open(a.valid_from))["checks"]
    rng = np.random.default_rng(SEED)
    per = {}
    for m in MODELS:
        f = R / f"s1p_{m}_d{a.depth}.json"
        if not f.exists() or m not in V1o:
            continue
        j = json.load(open(f))
        if "t_total_s" not in j:
            continue
        valid = V1o[m]["valid"]
        vj = [t for t in JUDG if valid[t]]
        vc = [t for t in CTRL if valid[t]]
        z = {}
        for c in CONTRASTS:
            nul = np.load(R / f"s1p_null_{c}_{m}_d{a.depth}.npy")
            z[c] = {}
            for t in JUDG + CTRL:
                col = nul[:, COLS.index(t)]
                sg = -1.0 if t == "L" else 1.0
                z[c][t] = float(sg * (j["res"]["effects"][c][t] - col.mean()) / col.std(ddof=1))
        zJ = np.array([np.mean([z[c][t] for t in vj]) for c in CONTRASTS])
        zC = np.array([np.mean([z[c][t] for t in vc]) for c in CONTRASTS]) if vc else np.full(len(CONTRASTS), np.nan)
        X = np.column_stack([zstd([j["ratings"][c][p]["mean"] for c in CONTRASTS]) for p in PRED])
        dig = [t for t in ("U_bright", "U_fam")]
        Lbr = {c: z[c]["L"] - np.mean([z[c][t] for t in vj if t != "L"]) for c in CONTRASTS} if "L" in vj else None
        per[m] = {"zJ": zJ, "zC": zC, "X": X, "z": z, "valid_judg": vj, "valid_ctrl": vc,
                  "ratings": {c: {p: j["ratings"][c][p]["mean"] for p in ("ext", "emo", "int", "val", "aro")}
                              for c in CONTRASTS},
                  "L_breakdown": Lbr, "digit_ctrl_z": {c: float(np.mean([z[c][t] for t in dig])) for c in CONTRASTS}}
    ms = list(per)
    out = {"models": ms, "n_perm": a.n_perm, "seed": SEED}
    if len(ms) < 2:
        out["status"] = "not testable"
        (R / "s1p_pooled_tests.json").write_text(json.dumps(out, indent=1))
        print(out)
        return

    tests = {}
    for k, p in enumerate(PRED):
        obs = sum(ols(per[m]["zJ"], per[m]["X"])[0][k] for m in ms)
        nul = sum(fl_coef_null(per[m]["zJ"], per[m]["X"], k, rng, a.n_perm) for m in ms)
        side = "upper" if p == "ext" else "two"
        if side == "upper":
            pv = (1 + np.sum(nul >= obs)) / (1 + a.n_perm)
        else:
            pv = (1 + np.sum(np.abs(nul - nul.mean()) >= abs(obs - nul.mean()))) / (1 + a.n_perm)
        tests[f"H-{p}"] = {"obs": float(obs), "null_mean": float(nul.mean()), "null_sd": float(nul.std(ddof=1)),
                           "p": float(pv), "side": side,
                           "per_model": {m: float(ols(per[m]["zJ"], per[m]["X"])[0][k]) for m in ms}}
    is_emo = np.array([c in EMOTION for c in CONTRASTS])

    def cat_stat(ys, labels):
        return float(np.mean([y[lab].mean() - y[~lab].mean() for y, lab in zip(ys, labels)]))

    for name, key in (("H-cat", "zJ"), ("H-sel", "sel")):
        ys = []
        for m in ms:
            y = per[m]["zJ"] if key == "zJ" else per[m]["zJ"] - per[m]["zC"]
            _, fit = ols(y, per[m]["X"][:, [0]])
            ys.append(y - fit)
        obs = cat_stat(ys, [is_emo] * len(ms))
        nul = np.array([cat_stat(ys, [rng.permutation(is_emo) for _ in ms]) for _ in range(a.n_perm)])
        pv = (1 + np.sum(np.abs(nul) >= abs(obs))) / (1 + a.n_perm)
        tests[name] = {"obs": obs, "null_sd": float(nul.std(ddof=1)), "p": float(pv), "side": "two"}
    out["tests"] = tests
    out["descriptive"] = {m: {"zJ": dict(zip(CONTRASTS, per[m]["zJ"].tolist())),
                              "zC": dict(zip(CONTRASTS, per[m]["zC"].tolist())),
                              "ratings": per[m]["ratings"], "z": per[m]["z"],
                              "valid_judg": per[m]["valid_judg"], "valid_ctrl": per[m]["valid_ctrl"],
                              "L_breakdown": per[m]["L_breakdown"], "digit_ctrl_z": per[m]["digit_ctrl_z"],
                              "pred_corr": np.corrcoef(per[m]["X"].T).tolist()} for m in ms}
    (R / "s1p_pooled_tests.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    main()
