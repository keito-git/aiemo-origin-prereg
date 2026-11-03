"""Stage 1o registered tests (CPU only; reads s1o_rating.py outputs). Design: 02_設計/11_段階1o_読み出し不変性_設計.
z_t(v) = (effect_t(v) - mean of the null) / sd of the null, per model and readout t; the null of int_neg is pair_neg,
of int_neu pair_neu (the same index j pairs the two). L is sign-reversed (sanction direction = up for every judgment
readout). The null of every statistic = the same statistic computed on null draw j (z-standardized with the same
mean and sd), j = 1..n_pair, within a model and summed over models with the same j.
Readout validity (no steering, per model; scores per scenario x severity):
  S       severity effect > 1, frac_pos >= .8
  HARSH   severity effect > 1, frac_pos >= .8, rho(., S) > .7
  L       severity effect < -.5, frac_neg >= .8, rho(., S) < -.5
  CAT_S   severity effect > 1, rotation consistency > .8, rho(., S) > .7
  CAT_S2  severity effect > .5, rotation consistency > .8, rho(., S) > .7  (values set on probe 3, K scenarios;
          confirmed on probe 4, L scenarios; sensitivity: > .7 and > 1.0)
  PAIR    severity effect > 1 (log-odds), frac_pos >= .8, rho(., S) > .7  (order consistency reported only)
  U_bright, U_fam   |severity effect| < 1.2, rho(mild, severe) > .5, |rho(., S)| < .3   (Stage 1n)
  CAT_U, CAT_R      |severity effect| < .5, rotation consistency > .5, |rho(., S)| < .3
  and, for every readout, the gradient positive control (necessary condition): one-sided p < .05 against the
  pair_neg nulls on the even scenarios.
Model entry: >= 3 valid judgment readouts with >= 1 digit (S, HARSH, L) and >= 1 choice (CAT_S, CAT_S2, PAIR);
  >= 1 valid control; anger check (Stage 1n anger_ok). Fewer than 2 entered models: "not testable".
  The neutral check is a property of the meaB sentences, fixed before registration on the six-model mean of probe 2
  (emotion difference .51 < 0.5 x 2.84); the per-model neu_ok is reported only.
Primary I1 (one-sided, upper): sum over entered models of the min over valid judgment readouts of z(int_neg)
  (cross-model worst-readout statistic; not an intersection-union test, see the descriptive IUT below).
Secondary (no correction): I2 (two-sided) sum of zmin(int_neg) - zmin(int_neu); I2 without OLMo; I3 (upper) sum of
  mean z over valid judgment readouts - mean z over valid controls (int_neg); I4 (upper) sum of min z over valid
  CAT_S2 and PAIR (models with at least one of the two valid); I1 without L; I1 with PAIR from pairs 23 and 34 only;
  K1' (upper) sum of z of the shared judgment-readout residual of the gradient split (int_neg, null pair_neg).
Descriptive: per model IUT p (max over valid judgment readouts of the one-sided p of z(int_neg)), the number of models
  with IUT p < .05, all (model, readout) pairs with z < 0; z tables for int_neg, int_neu, int_pos, pc1_neg; the 10 x 10
  positive-control matrix; lexical cos with its null rank; CAT_S2 validity sensitivity.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from s1o_rating import CTRL, COLS, JUDG, READ

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
DIG, CHO = ("S", "HARSH", "L"), ("CAT_S", "CAT_S2", "PAIR")
SGN = {t: (-1.0 if t == "L" else 1.0) for t in READ}


def ok(x, thr=0.5, frac=0.75):
    return x["mean"] > thr and x["frac_pos"] >= frac


def summ(x, xS):
    x, xS = np.asarray(x), np.asarray(xS)
    n = len(x) // 2
    d = x[n:] - x[:n]
    return {"severity_effect": float(d.mean()), "frac_pos": float((d > 0).mean()), "frac_neg": float((d < 0).mean()),
            "rho_mild_severe": float(spearmanr(x[:n], x[n:])[0]), "rho_with_S": float(spearmanr(x, xS)[0])}


def rot_cons(E):
    E = np.asarray(E)
    return float(np.mean([spearmanr(E[i], E[j])[0] for i, j in combinations(range(len(E)), 2)]))


def validity(sc, cat_s2_thr=0.5):
    s = {t: summ(sc[t], sc["S"]) for t in READ}
    for t in ("CAT_S", "CAT_S2", "CAT_U", "CAT_R"):
        s[t]["rot_consistency"] = rot_cons(sc[f"{t}_by_rot"])
    P = np.asarray(sc["PAIR_by_rot"])                 # rows: 12a, 12b, 23a, 23b, 34a, 34b
    s["PAIR"]["order_consistency"] = float(np.mean([spearmanr(P[2 * k], P[2 * k + 1])[0] for k in range(3)]))
    u = lambda x: abs(x["severity_effect"]) < 1.2 and x["rho_mild_severe"] > .5 and abs(x["rho_with_S"]) < .3
    c = lambda x: abs(x["severity_effect"]) < .5 and x["rot_consistency"] > .5 and abs(x["rho_with_S"]) < .3
    v = {"S": s["S"]["severity_effect"] > 1 and s["S"]["frac_pos"] >= .8,
         "HARSH": s["HARSH"]["severity_effect"] > 1 and s["HARSH"]["frac_pos"] >= .8 and s["HARSH"]["rho_with_S"] > .7,
         "L": s["L"]["severity_effect"] < -.5 and s["L"]["frac_neg"] >= .8 and s["L"]["rho_with_S"] < -.5,
         "CAT_S": (s["CAT_S"]["severity_effect"] > 1 and s["CAT_S"]["rot_consistency"] > .8
                   and s["CAT_S"]["rho_with_S"] > .7),
         "CAT_S2": (s["CAT_S2"]["severity_effect"] > cat_s2_thr and s["CAT_S2"]["rot_consistency"] > .8
                    and s["CAT_S2"]["rho_with_S"] > .7),
         "PAIR": s["PAIR"]["severity_effect"] > 1 and s["PAIR"]["frac_pos"] >= .8 and s["PAIR"]["rho_with_S"] > .7,
         "U_bright": u(s["U_bright"]), "U_fam": u(s["U_fam"]), "CAT_U": c(s["CAT_U"]), "CAT_R": c(s["CAT_R"])}
    return s, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)
    tagof = lambda m: f"{m}_anger_d{a.depth}"
    J, NUL, PRJ = {}, {}, {}
    for m in MODELS:
        f = R / f"s1o_{tagof(m)}.json"
        if not f.exists():
            continue
        j = json.load(open(f))
        if "t_total_s" not in j:
            continue
        J[m] = j
        NUL[m] = {nm: np.load(R / f"s1o_null_{nm}_{tagof(m)}.npy") for nm in ("pair_neg", "pair_neu", "int_pos",
                                                                            "pc1_neg")}
        PRJ[m] = np.load(R / f"s1o_proj_null_pair_neg_{tagof(m)}.npy")
    NFILE = {"int_neg": "pair_neg", "int_neu": "pair_neu", "int_pos": "int_pos", "pc1_neg": "pc1_neg"}

    def z_obs_null(m, nm, col):
        """z of the observed effect and of every null draw (sign-aligned), column col."""
        nul = NUL[m][NFILE[nm]][:, COLS.index(col)]
        obs = J[m]["res"]["effects"][nm][col]
        sg = SGN.get(col, 1.0) if not col.startswith("PAIR_") else 1.0
        mu, sd = nul.mean(), nul.std(ddof=1)
        return sg * (obs - mu) / sd, sg * (nul - mu) / sd

    def pair_col(m, nm, cols):
        """PAIR from a subset of adjacent pairs: mean of the per-pair effects."""
        nul = NUL[m][NFILE[nm]][:, [COLS.index(c) for c in cols]].mean(1)
        obs = float(np.mean([J[m]["res"]["effects"][nm][c] for c in cols]))
        mu, sd = nul.mean(), nul.std(ddof=1)
        return (obs - mu) / sd, (nul - mu) / sd

    checks, entered = {}, []
    for m, j in J.items():
        s, v = validity(j["base"]["scores"])
        # gradient positive control on the even scenarios against pair_neg (necessary condition)
        pos_p = {}
        for t in READ:
            nul = NUL[m]["pair_neg"][:, COLS.index(f"even_{t}")]
            pos_p[t] = float((np.sum(nul >= j["pos"]["obs"][t]) + 1) / (len(nul) + 1))
        valid = {t: bool(v[t] and pos_p[t] < .05) for t in READ}
        mc = j["manipulation"]
        anger_ok = ok(mc["intensity_anger_L_minus_anger_lo"]) and ok(mc["extremity_anger_L_minus_anger_lo"])
        neu_ok = (ok(mc["extremity_meaB_hi_minus_meaB_lo"])
                  and mc["emotion"]["neu_hi_minus_lo"] < 0.5 * mc["emotion"]["anger_L_minus_lo"])
        vj = [t for t in JUDG if valid[t]]
        enter = (len(vj) >= 3 and any(t in DIG for t in vj) and any(t in CHO for t in vj)
                 and any(valid[t] for t in CTRL) and anger_ok)
        _, v10 = validity(j["base"]["scores"], 1.0)
        _, v07 = validity(j["base"]["scores"], 0.7)
        checks[m] = {"summ": s, "valid_noSteer": v, "pos_p": pos_p, "valid": valid, "anger_ok": anger_ok,
                     "neu_ok_reported_only": neu_ok, "emotion": mc["emotion"], "enter": enter,
                     "CAT_S2_valid_thr07": v07["CAT_S2"], "CAT_S2_valid_thr10": v10["CAT_S2"],
                     "check_1n": j.get("check_1n"), "null_pair": j["res"]["null"]["pair"]}
        if enter:
            entered.append(m)

    def zmin(m, nm, readouts, pair_cols=None):
        zs, ns = [], []
        for t in readouts:
            if t == "PAIR" and pair_cols:
                zo, zn = pair_col(m, nm, pair_cols)
            else:
                zo, zn = z_obs_null(m, nm, t)
            zs.append(zo)
            ns.append(zn)
        return min(zs), np.min(np.stack(ns), 0)

    def test(models, fn, side):
        if len(models) < 2:
            return {"models": list(models), "not_testable": True}
        obs, nul = 0.0, 0.0
        for m in models:
            o, n = fn(m)
            obs, nul = obs + o, nul + n
        pu = float((np.sum(nul >= obs) + 1) / (len(nul) + 1))
        pl = float((np.sum(nul <= obs) + 1) / (len(nul) + 1))
        p = {"upper": pu, "two": min(1.0, 2 * min(pu, pl))}[side]
        return {"models": list(models), "obs": float(obs), "null_mean": float(nul.mean()),
                "null_sd": float(nul.std(ddof=1)), "p": p, "side": side, "significant": p < .05}

    VJ = lambda m: [t for t in JUDG if checks[m]["valid"][t]]
    VC = lambda m: [t for t in CTRL if checks[m]["valid"][t]]
    out = {"entered": entered, "checks": checks}
    out["primary"] = {"I1": test(entered, lambda m: zmin(m, "int_neg", VJ(m)), "upper")}

    def i2(m):
        o1, n1 = zmin(m, "int_neg", VJ(m))
        o2, n2 = zmin(m, "int_neu", VJ(m))
        return o1 - o2, n1 - n2

    def i3(m):
        zj = [z_obs_null(m, "int_neg", t) for t in VJ(m)]
        zc = [z_obs_null(m, "int_neg", t) for t in VC(m)]
        return (np.mean([x[0] for x in zj]) - np.mean([x[0] for x in zc]),
                np.mean([x[1] for x in zj], 0) - np.mean([x[1] for x in zc], 0))

    lex_free = lambda m: [t for t in ("CAT_S2", "PAIR") if checks[m]["valid"][t]]

    def k1(m):
        k = J[m]["grad"]["proj_cols"].index("shared")
        nul = PRJ[m][:, k]
        obs = J[m]["grad"]["proj_dirs"]["int_neg"]["shared"]
        return (obs - nul.mean()) / nul.std(ddof=1), (nul - nul.mean()) / nul.std(ddof=1)

    out["secondary"] = {
        "I2_neg_minus_neu": test(entered, i2, "two"),
        "I2_without_OLMo": test([m for m in entered if not m.startswith("OLMo")], i2, "two"),
        "I3_selectivity": test(entered, i3, "upper"),
        "I4_CAT_S2_PAIR": test([m for m in entered if lex_free(m)], lambda m: zmin(m, "int_neg", lex_free(m)), "upper"),
        "I1_without_L": test(entered, lambda m: zmin(m, "int_neg", [t for t in VJ(m) if t != "L"]), "upper"),
        "I1_PAIR_23_34": test(entered, lambda m: zmin(m, "int_neg", VJ(m), ("PAIR_23", "PAIR_34")), "upper"),
        "K1_shared_residual": test(entered, k1, "upper"),
    }
    # descriptive
    zt = {nm: {m: {t: float(z_obs_null(m, nm, t)[0]) for t in READ} for m in J} for nm in NFILE}
    iut = {}
    for m in entered:
        ps = []
        for t in VJ(m):
            zo, zn = z_obs_null(m, "int_neg", t)
            ps.append(float((np.sum(zn >= zo) + 1) / (len(zn) + 1)))
        iut[m] = max(ps)
    out["descriptive"] = {
        "z": zt, "IUT_p": iut, "IUT_n_below_05": int(sum(p < .05 for p in iut.values())),
        "negative_pairs_int_neg": [(m, t, zt["int_neg"][m][t]) for m in entered for t in VJ(m)
                                   if zt["int_neg"][m][t] < 0],
        "pos_matrix_even": {m: J[m]["pos"]["matrix_even"] for m in J},
        "lexical": {m: {"cos_dirs": J[m]["lexical"]["cos_dirs"],
                        "int_neg_abs_cos_rank_in_null": {
                            t: float(np.mean(np.abs(np.load(R / f"s1o_lex_null_pair_neg_{tagof(m)}.npy")[:, i])
                                             >= abs(J[m]["lexical"]["cos_dirs"]["int_neg"][t])))
                            for i, t in enumerate(J[m]["lexical"]["cos_dirs"]["int_neg"])}} for m in J},
        "grad_split": {m: J[m]["grad"]["split"] for m in J},
    }
    (R / "s1o_pooled_tests.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"entered": entered, "primary": out["primary"],
                      "secondary": {k: {kk: vv for kk, vv in v.items() if kk != "models"}
                                    for k, v in out["secondary"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
