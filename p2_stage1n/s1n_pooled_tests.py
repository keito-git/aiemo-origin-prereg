"""Stage 1n registered tests (CPU only; reads s1n_rating.py outputs).

D_X,Y(v) = dslope_X(v) - dslope_Y(v) for questions X, Y (S sanction, C location, L leniency, HARSH, U_bright, U_fam,
U_size). Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a
model). int_neg and int_neu use the paired null (s1n_null_pair_neg / _neu).
Gate (per model), as in Stage 1m: pos_82 or pos_sev p < 0.05; S severity effect > 1 and frac >= 0.8; C rho with
location > 0.5 and |C severity effect| < 1; L severity effect < -0.5, frac negative >= 0.8 and rho(S, L) < -0.5;
valence check > 2. Added for Stage 1n: U_bright validity (|severity effect| < 1.2, rho(mild, severe) > 0.5,
|rho(U_bright, S)| < 0.3) and pos_U82 p < 0.05 (the U_bright readout responds to a direction).
Anger checks and neutral checks as in Stage 1m.
Primary (one test, one-sided, alpha 0.05):
  N1 = sum over gated models with the anger checks of D_S,U_bright(int_neg), upper: the anger intensity direction
       raises the sanction rating more than a degree question about the place (against the strong form of C').
Secondary (no correction): see SECONDARY below; the U_size - U_bright gradient tests the weak form of C'.
A test needs at least two models; if only one model is testable, its p is doubled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import p_lower, p_upper

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
TASKS = ("S", "C", "L", "HARSH", "U_bright", "U_fam", "U_size")
COLS = tuple(f"dE_{t}" for t in TASKS) + tuple(f"dslope_{t}" for t in TASKS)
NULL_FILE = {"int_neg": "pair_neg", "int_neu": "pair_neu", "int_pos": "int_pos", "pc1_neg": "pc1_neg"}


def ok(x, thr=0.5, frac=0.75):
    return x["mean"] > thr and x["frac_pos"] >= frac


def u_valid(t):
    return abs(t["severity_effect"]) < 1.2 and t["rho_mild_severe"] > 0.5 and abs(t["rho_with_S"]) < 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def jj(m):
        f = R / f"s1n_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, x, y, n):
        """(observed, null draws) of dslope_x - dslope_y (y None: dslope_x alone) for direction n in model m."""
        e = jj(m)["readout"]["R9n"]["effects"][n]
        z = np.load(R / f"s1n_null_{NULL_FILE[n]}_{m}_anger_d{a.depth}.npy")
        col = lambda t: z[:, COLS.index(f"dslope_{t}")]
        if y is None:
            return e[f"dslope_{x}"], col(x)
        return e[f"dslope_{x}"] - e[f"dslope_{y}"], col(x) - col(y)

    gate, checks = [], {}
    for m in MODELS:
        j = jj(m)
        if not j or "t_total_s" not in j or "readout" not in j:
            continue
        r, mc, tk = j["readout"]["R9n"], j["manipulation"], j["checks"]["tasks"]
        c = {"S_severity_effect": tk["S"]["severity_effect"], "S_frac_pos": tk["S"]["frac_pos"],
             "C_severity_effect": tk["C"]["severity_effect"], "C_rho_location": tk["C"]["rho_location"],
             "L_severity_effect": tk["L"]["severity_effect"], "L_frac_neg": tk["L"]["frac_neg"],
             "rho_S_L": tk["L"]["rho_with_S"], "U": {u: tk[u] for u in ("U_bright", "U_fam", "U_size")}}
        c |= {f"{k}_p": r["null"][k]["p_one"] for k in ("pos_82", "pos_sev", "pos_U82")}
        c["valence_pos_minus_neg"] = mc["valence_pos_minus_neg"]["mean"]
        c["anger_ok"] = ok(mc["intensity_anger_L_minus_anger_lo"]) and ok(mc["extremity_anger_L_minus_anger_lo"])
        c["joy_ok"] = ok(mc["intensity_joy_L_minus_joy_lo"]) and ok(mc["extremity_joy_L_minus_joy_lo"])
        c["neu_ok"] = (ok(mc["extremity_neu_hi_minus_neu_lo"])
                       and mc["emotion"]["neu_hi_minus_lo"] < 0.5 * mc["emotion"]["anger_L_minus_lo"])
        c["U_valid"] = {u: u_valid(tk[u]) for u in ("U_bright", "U_fam", "U_size")}
        c["pair"] = r["null"]["pair"]
        c["cos_with_1m"] = j["cos_with_1m"]
        c["pass"] = ((c["pos_82_p"] < 0.05 or c["pos_sev_p"] < 0.05) and c["S_severity_effect"] > 1
                     and c["S_frac_pos"] >= 0.8 and c["C_rho_location"] > 0.5
                     and abs(c["C_severity_effect"]) < 1 and c["L_severity_effect"] < -0.5
                     and c["L_frac_neg"] >= 0.8 and c["rho_S_L"] < -0.5 and c["valence_pos_minus_neg"] > 2
                     and c["U_valid"]["U_bright"] and c["pos_U82_p"] < 0.05)
        checks[m] = c
        if c["pass"]:
            gate.append(m)
    g1 = [m for m in gate if checks[m]["anger_ok"]]
    gn = [m for m in gate if checks[m]["neu_ok"]]
    g2 = [m for m in g1 if checks[m]["neu_ok"]]
    gp = [m for m in gate if checks[m]["joy_ok"]]
    gu = lambda g, u: [m for m in g if checks[m]["U_valid"][u]]

    def stat(models, terms):
        if len(models) == 0:
            return {"models": [], "not_testable": True}
        obs, nul = 0.0, 0.0
        for m in models:
            for sign, x, y, n in terms:
                o, z = src(m, x, y, n)
                obs, nul = obs + sign * o, nul + sign * z
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        k = 2.0 if len(models) == 1 else 1.0
        return {"models": list(models), "obs": float(obs), "null_mean": float(np.mean(nul)),
                "null_sd": float(np.std(nul, ddof=1)), "null_q05": float(np.quantile(nul, 0.05)),
                "null_q95": float(np.quantile(nul, 0.95)), "p_upper": min(1.0, k * pu), "p_lower": min(1.0, k * pl),
                "p_two": min(1.0, 2 * k * min(pu, pl))}

    out = {"gate": gate, "gate_N1": g1, "gate_neu": gn, "gate_neg_neu": g2, "gate_joy": gp, "checks": checks}
    if g1:
        n1 = stat(g1, [(1, "S", "U_bright", "int_neg")])
        n1["p_registered"] = n1["p_upper"]
        n1["significant"] = n1["p_upper"] < 0.05
        out["primary"] = {"N1_S_minus_Ubright_int_neg_upper": n1}
    else:
        out["primary"] = {"not_testable": True}
    if gate:
        SECONDARY = {
            # other degree questions (models where that U is valid), upper
            "N1_S_minus_Ufam_int_neg_upper": stat(gu(g1, "U_fam"), [(1, "S", "U_fam", "int_neg")]),
            "N1_S_minus_Usize_int_neg_upper": stat(g1, [(1, "S", "U_size", "int_neg")]),
            # weak C': the question close in meaning to the neutral pairs (large / enormous) moves more, upper
            "grad_Usize_minus_Ubright_int_neu_upper": stat(gn, [(1, "U_size", "U_bright", "int_neu")]),
            "grad_Usize_minus_Ubright_int_neg_two": stat(g1, [(1, "U_size", "U_bright", "int_neg")]),
            # neutral intensity and the anger - neutral difference (paired null)
            "N1_int_neu_upper": stat(gn, [(1, "S", "U_bright", "int_neu")]),
            "N1_int_neg_minus_int_neu_two": stat(g2, [(1, "S", "U_bright", "int_neg"), (-1, "S", "U_bright", "int_neu")]),
            # leniency against the degree question, lower
            "N2_L_minus_Ubright_int_neg_lower": stat(g1, [(1, "L", "U_bright", "int_neg")]),
            "N2_L_minus_Ubright_int_neu_two": stat(gn, [(1, "L", "U_bright", "int_neu")]),
            # paraphrase of S, upper
            "N1_HARSH_minus_Ubright_int_neg_upper": stat(g1, [(1, "HARSH", "U_bright", "int_neg")]),
            # Stage 1m J_S again (same items), upper, and whether U itself moves against C (two-sided)
            "JS_S_minus_C_int_neg_upper": stat(g1, [(1, "S", "C", "int_neg")]),
            "Ubright_minus_C_int_neg_two": stat(g1, [(1, "U_bright", "C", "int_neg")]),
            "Ubright_minus_C_int_neu_two": stat(gn, [(1, "U_bright", "C", "int_neu")]),
            "Usize_minus_C_int_neu_two": stat(gn, [(1, "U_size", "C", "int_neu")]),
            # other directions, two-sided
            "N1_int_pos_two": stat(gp, [(1, "S", "U_bright", "int_pos")]),
            "N1_pc1_neg_two": stat(gate, [(1, "S", "U_bright", "pc1_neg")]),
        }
        out["secondary"] = SECONDARY
        # descriptive: summed dslope per question minus the summed null mean, over the gated models
        out["split_minus_null_mean"] = {
            n: {t: float(sum(src(m, t, None, n)[0] - src(m, t, None, n)[1].mean() for m in gate)) for t in TASKS}
            for n in NULL_FILE}
    out["descriptive_single"] = {m: {"effects": jj(m)["readout"]["R9n"]["effects"], "cos": jj(m)["cos"]}
                                 for m in checks}
    (R / "s1n_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "gate_N1", "gate_neu", "gate_neg_neu", "primary")}
                     | {"secondary": out.get("secondary"), "split_minus_null_mean": out.get("split_minus_null_mean")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
