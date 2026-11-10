"""Stage 1q pooled tests (design 02_設計/13_段階1q_整列の前向き検証_設計_draft_20261110.md; registered before any
Stage 1q steered run of a planned model).
Per model and contrast c (the twelve new magnitude contrasts):
  projection z = (alpha * D_c @ shared_A - null mean) / null SD, where shared_A is the shared judgment gradient on the
  A items (L01-L30) and the null is the n_null variance-matched random directions of c (s1q_proj_null_*);
  zJ_B = mean over the valid judgment readouts of the z of the B-item effect (L31-L60) against the same nulls
  (L sign-reversed). Readout validity is taken from the Stage 1o pooled tests.
  H-align (primary, one-sided upper): mean over models of atanh(r), r = Pearson r of projection z and zJ_B over the
    contrasts of that model. Null: the projection z permuted over contrasts within each model, n_perm draws.
  H-val2 (secondary, one-sided lower): sum over models of [mean of the unpleasant - mean of the pleasant contrasts] of
    zJ_B after regressing zJ_B on the standardized ext rating within model. Null: group labels permuted within model.
  p = (1 + #{null at least as extreme in the predicted direction}) / (1 + n_perm); seed 9150.
Descriptive: per-readout alignment r (each judgment readout against the projection z); the OLS of zJ_B on ext, emo,
val within model; zJ_B, zC_B and the ratings per contrast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1o_rating import COLS, CTRL, JUDG
from s1q_rating import CONTRASTS, PLEASANT, UNPLEASANT

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
SEED = 9150


def zstd(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    return b[1:], X1 @ b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--valid_from", required=True, help="Stage 1o s1o_pooled_tests.json")
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--all_valid", action="store_true", help="code-path dry run only: treat every readout as valid")
    a = ap.parse_args()
    R = Path(a.res)
    V1o = json.load(open(a.valid_from))["checks"]
    rng = np.random.default_rng(SEED)
    per = {}
    for m in a.models.split(","):
        f = R / f"s1q_{m}_d{a.depth}.json"
        if not f.exists() or (m not in V1o and not a.all_valid):
            continue
        j = json.load(open(f))
        if "t_total_s" not in j:
            continue
        valid = V1o[m]["valid"] if m in V1o else {t: True for t in JUDG + CTRL}
        vj = [t for t in JUDG if valid[t]]
        vc = [t for t in CTRL if valid[t]]
        pred = json.load(open(R / f"s1q_pred_{m}_d{a.depth}.json"))
        k0 = pred["proj_cols"].index("shared")
        z, pz = {}, {}
        for c in CONTRASTS:
            nul = np.load(R / f"s1q_null_{c}_{m}_d{a.depth}.npy")
            z[c] = {}
            for t in JUDG + CTRL:
                col = nul[:, COLS.index(t)]
                sg = -1.0 if t == "L" else 1.0
                z[c][t] = float(sg * (j["res"]["effects_B"][c][t] - col.mean()) / col.std(ddof=1))
            pn = np.load(R / f"s1q_proj_null_{c}_{m}_d{a.depth}.npy")[:, k0]
            pz[c] = float((pred["proj_dirs"][c]["shared"] - pn.mean()) / pn.std(ddof=1))
        zJ = np.array([np.mean([z[c][t] for t in vj]) for c in CONTRASTS])
        zC = np.array([np.mean([z[c][t] for t in vc]) for c in CONTRASTS]) if vc else np.full(len(CONTRASTS), np.nan)
        P = np.array([pz[c] for c in CONTRASTS])
        X = np.column_stack([zstd([j["ratings"][c][p]["mean"] for c in CONTRASTS]) for p in ("ext", "emo", "val")])
        per[m] = {"zJ": zJ, "zC": zC, "P": P, "X": X, "z": z, "vj": vj, "vc": vc,
                  "ratings": {c: {p: j["ratings"][c][p]["mean"] for p in ("ext", "emo", "int", "val", "aro")}
                              for c in CONTRASTS}}
    ms = list(per)
    out = {"models": ms, "n_perm": a.n_perm, "seed": SEED}
    if len(ms) < 2:
        out["status"] = "not testable"
        (R / "s1q_pooled_tests.json").write_text(json.dumps(out, indent=1))
        print(out)
        return

    tests = {}
    r_obs = {m: float(np.corrcoef(per[m]["P"], per[m]["zJ"])[0, 1]) for m in ms}
    obs = float(np.mean([np.arctanh(r_obs[m]) for m in ms]))
    nul = np.array([np.mean([np.arctanh(np.corrcoef(rng.permutation(per[m]["P"]), per[m]["zJ"])[0, 1]) for m in ms])
                    for _ in range(a.n_perm)])
    tests["H-align"] = {"obs": obs, "r_per_model": r_obs, "null_mean": float(nul.mean()),
                        "null_sd": float(nul.std(ddof=1)), "p": float((1 + np.sum(nul >= obs)) / (1 + a.n_perm)),
                        "side": "upper"}

    is_un = np.array([c in UNPLEASANT for c in CONTRASTS])
    assert set(np.array(CONTRASTS)[~is_un]) == set(PLEASANT)
    ys = []
    for m in ms:
        _, fit = ols(per[m]["zJ"], per[m]["X"][:, [0]])
        ys.append(per[m]["zJ"] - fit)

    def val_stat(labels):
        return float(sum(y[lab].mean() - y[~lab].mean() for y, lab in zip(ys, labels)))

    obs = val_stat([is_un] * len(ms))
    nul = np.array([val_stat([rng.permutation(is_un) for _ in ms]) for _ in range(a.n_perm)])
    tests["H-val2"] = {"obs": obs, "per_model": {m: float(y[is_un].mean() - y[~is_un].mean()) for m, y in zip(ms, ys)},
                       "null_sd": float(nul.std(ddof=1)), "p": float((1 + np.sum(nul <= obs)) / (1 + a.n_perm)),
                       "side": "lower"}
    out["tests"] = tests
    out["descriptive"] = {m: {"zJ_B": dict(zip(CONTRASTS, per[m]["zJ"].tolist())),
                              "zC_B": dict(zip(CONTRASTS, per[m]["zC"].tolist())),
                              "proj_z": dict(zip(CONTRASTS, per[m]["P"].tolist())),
                              "align_r_by_readout": {t: float(np.corrcoef(per[m]["P"], [per[m]["z"][c][t]
                                                                                       for c in CONTRASTS])[0, 1])
                                                     for t in per[m]["vj"]},
                              "ols_ext_emo_val": ols(per[m]["zJ"], per[m]["X"])[0].tolist(),
                              "ratings": per[m]["ratings"], "z": per[m]["z"],
                              "valid_judg": per[m]["vj"], "valid_ctrl": per[m]["vc"]} for m in ms}
    (R / "s1q_pooled_tests.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    main()
