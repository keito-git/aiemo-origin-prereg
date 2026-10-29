"""Stage 1m registered tests (CPU only; reads s1m_rating.py outputs).

J_S(v) = dslope_S(v) - dslope_C(v)   sanction question minus location question (Stage 1k K3, Stage 1l J)
J_L(v) = dslope_L(v) - dslope_C(v)   leniency question minus location question (higher L digit = milder)
Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model).
int_neg and int_neu use the paired null (s1m_null_pair_neg / _neu, cos matched to cos(int_neg, int_neu)).
Gate (per model): pos_82 or pos_sev p < 0.05; S severity effect > 1 and frac >= 0.8; C rho with location > 0.5 and
|C severity effect| < 1; L severity effect < -0.5, frac negative >= 0.8 and Spearman(S, L) < -0.5; valence check > 2.
Manipulation checks: anger intensity (Q_INT) and anger extremity (Q_EXT) high minus low > 0.5, positive in >= 75%;
neutral extremity (Q_EXT) high minus low > 0.5, positive in >= 75%; neutral emotion (Q_EMO) high minus low < half
of the anger emotion difference (the probe showed that 8B models rate extreme non-emotional sentences as somewhat
emotional, so an absolute bound would exclude most models).
Primary (Holm over two, one-sided):
  M1 = sum over gated models with the anger checks of J_L(int_neg), lower: the intensity direction lowers the
       leniency rating, so the judgment becomes harsher (not "more of whatever is asked").
  M2 = sum over gated models with the anger and neutral checks of [J_S(int_neg) - J_S(int_neu)], upper: emotional
       intensity raises the sanction more than non-emotional intensity.
Secondary (no correction): sum J_S(int_neg) (upper); sum dslope_L(int_neg) (lower); sum J_L(int_pos), sum J_S(int_neu),
  sum J_L(int_neu), sum [J_L(int_neg) - J_L(int_neu)] (paired), M2 over all models passing the neutral extremity
  check only (upper), sum J_S/J_L of int_com, int_dif, pc1_neg, pc1_pos
  (all two-sided).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import holm, p_lower, p_upper

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
COLS = ("dE_S", "dE_C", "dE_L", "dslope_S", "dslope_C", "dslope_L")
NULL_FILE = {"int_neg": "pair_neg", "int_neu": "pair_neu", "int_pos": "int_pos", "int_com": "int_com",
             "int_dif": "int_dif", "pc1_neg": "pc1_neg", "pc1_pos": "pc1_pos"}


def ok(x, thr=0.5, frac=0.75):
    return x["mean"] > thr and x["frac_pos"] >= frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def jj(m):
        f = R / f"s1m_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, c, n):
        """(observed, null draws) of column c ('JS', 'JL' or a COLS name) for direction n in model m."""
        e = jj(m)["readout"]["R9m"]["effects"][n]
        z = np.load(R / f"s1m_null_{NULL_FILE[n]}_{m}_anger_d{a.depth}.npy")
        col = lambda k: z[:, COLS.index(k)]
        if c == "JS":
            return e["dslope_S"] - e["dslope_C"], col("dslope_S") - col("dslope_C")
        if c == "JL":
            return e["dslope_L"] - e["dslope_C"], col("dslope_L") - col("dslope_C")
        return e[c], col(c)

    gate, checks = [], {}
    for m in MODELS:
        j = jj(m)
        if not j or "t_total_s" not in j or "readout" not in j:
            continue
        r, mc, ch = j["readout"]["R9m"], j["manipulation"], j["checks"]
        c = {k: ch[k] for k in ("severity_effect", "severity_frac_pos", "C_severity_effect", "C_rho_location",
                                "L_severity_effect", "L_severity_frac_neg", "rho_S_L")}
        c |= {f"{k}_p": r["null"][k]["p_one"] for k in ("pos_82", "pos_sev")}
        c["valence_pos_minus_neg"] = mc["valence_pos_minus_neg"]["mean"]
        c["anger_ok"] = ok(mc["intensity_anger_L_minus_anger_lo"]) and ok(mc["extremity_anger_L_minus_anger_lo"])
        c["joy_ok"] = ok(mc["intensity_joy_L_minus_joy_lo"]) and ok(mc["extremity_joy_L_minus_joy_lo"])
        c["neu_ok"] = (ok(mc["extremity_neu_hi_minus_neu_lo"])
                       and mc["emotion"]["neu_hi_minus_lo"] < 0.5 * mc["emotion"]["anger_L_minus_lo"])
        c["neu_ext_ok"] = ok(mc["extremity_neu_hi_minus_neu_lo"])
        c["neu_extremity"] = mc["extremity_neu_hi_minus_neu_lo"]["mean"]
        c["neu_emotion_diff"] = mc["emotion"]["neu_hi_minus_lo"]
        c["pair"] = r["null"]["pair"]
        c["cos_with_1l"] = j["cos_with_1l"]
        c["pass"] = ((c["pos_82_p"] < 0.05 or c["pos_sev_p"] < 0.05) and c["severity_effect"] > 1
                     and c["severity_frac_pos"] >= 0.8 and c["C_rho_location"] > 0.5
                     and abs(c["C_severity_effect"]) < 1 and c["L_severity_effect"] < -0.5
                     and c["L_severity_frac_neg"] >= 0.8 and c["rho_S_L"] < -0.5 and c["valence_pos_minus_neg"] > 2)
        checks[m] = c
        if c["pass"]:
            gate.append(m)
    g1 = [m for m in gate if checks[m]["anger_ok"]]
    g2 = [m for m in g1 if checks[m]["neu_ok"]]
    gp = [m for m in gate if checks[m]["joy_ok"]]

    def stat(models, terms):
        if not models:
            return {"models": [], "not_testable": True}
        obs, nul = 0.0, 0.0
        for m in models:
            for sign, c, n in terms:
                x, z = src(m, c, n)
                obs, nul = obs + sign * x, nul + sign * z
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": float(obs), "null_mean": float(np.mean(nul)),
                "null_sd": float(np.std(nul, ddof=1)), "null_q05": float(np.quantile(nul, 0.05)),
                "null_q95": float(np.quantile(nul, 0.95)), "p_upper": pu, "p_lower": pl,
                "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate": gate, "gate_M1": g1, "gate_M2": g2, "gate_joy": gp, "checks": checks}
    prim, p_one = {}, {}
    if len(g1) >= 2:
        prim["M1_JL_int_neg_lower"] = stat(g1, [(1, "JL", "int_neg")])
        p_one["M1_JL_int_neg_lower"] = prim["M1_JL_int_neg_lower"]["p_lower"]
    if len(g2) >= 2:
        prim["M2_JS_int_neg_minus_int_neu"] = stat(g2, [(1, "JS", "int_neg"), (-1, "JS", "int_neu")])
        p_one["M2_JS_int_neg_minus_int_neu"] = prim["M2_JS_int_neg_minus_int_neu"]["p_upper"]
    adj = holm(p_one) if p_one else {}
    for k in prim:
        prim[k]["p_one_registered"] = p_one[k]
        prim[k]["p_holm"] = min(1.0, adj[k] * (2 / len(prim)) if len(prim) < 2 else adj[k])   # Holm over two
        prim[k]["significant"] = prim[k]["p_holm"] < 0.05
    out["primary"] = prim or {"not_testable": True}
    if gate:
        out["secondary"] = {
            "JS_int_neg": stat(g1, [(1, "JS", "int_neg")]),
            "dslope_L_int_neg": stat(g1, [(1, "dslope_L", "int_neg")]),
            "JL_int_pos": stat(gp, [(1, "JL", "int_pos")]),
            "JS_int_neu": stat([m for m in gate if checks[m]["neu_ok"]], [(1, "JS", "int_neu")]),
            "JL_int_neu": stat([m for m in gate if checks[m]["neu_ok"]], [(1, "JL", "int_neu")]),
            "JL_int_neg_minus_int_neu": stat(g2, [(1, "JL", "int_neg"), (-1, "JL", "int_neu")]),
            "M2_all_extremity_ok": stat([m for m in g1 if checks[m]["neu_ext_ok"]],
                                        [(1, "JS", "int_neg"), (-1, "JS", "int_neu")]),
            **{f"{c}_{n}": stat(gate, [(1, c, n)]) for n in ("int_com", "int_dif", "pc1_neg", "pc1_pos")
               for c in ("JS", "JL")}}
    out["descriptive_single"] = {m: {"effects": jj(m)["readout"]["R9m"]["effects"], "cos": jj(m)["cos"],
                                     "manipulation": jj(m)["manipulation"]} for m in checks}
    (R / "s1m_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "gate_M1", "gate_M2", "gate_joy", "primary")}
                     | {"secondary": out.get("secondary")}, indent=1, default=float))


if __name__ == "__main__":
    main()
