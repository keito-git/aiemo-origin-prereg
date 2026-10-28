"""Stage 1l registered tests (CPU only; reads s1l_rating.py outputs).

Contrast J(v) = dslope_S(v) - dslope_C(v) (judgment-specific shift on the saturation-free metric, as K3 in Stage 1k).
Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model).
pc1_neg vs pc1_pos uses the paired null (s1l_null_pair_neg / _pos): per draw, two variance-matched random directions
whose cos equals cos(pc1_neg, pc1_pos).
Gate (per model, as Stage 1k): pos_82 or pos_sev p < 0.05; severity_effect > 1 and frac >= 0.8; C_rho_location > 0.5
and |C_severity_effect| < 1; valence check (pleasantness of the positive triple minus the negative triple) > 2.
Intensity check (per valence): mean rated strength high minus low > 0.5 and positive in >= 75% of the 12 pairs.
Primary (Holm over two, one-sided upper):
  L1 = sum over gated models of [J(pc1_neg) - J(pc1_pos)]: the judgment shift is larger for negative than for
       positive emotion.
  L2 = sum over gated models that pass the anger intensity check of J(int_neg): negative emotional intensity
       (furious vs irritated) shifts the judgment.
Secondary (no correction): L1e = sum [dE_S(pc1_neg) - dE_S(pc1_pos)] (upper); L1s = sum [dslope_S(pc1_neg) -
  dslope_S(pc1_pos)] (upper); sum J(pc1_neg) (upper; Stage 1k K3 on the paired null draws); sum J(pc1_pos) and
  sum dE_S(pc1_pos) (two-sided); sum J(int_pos) over models passing the joy intensity check (two-sided);
  sum dE_S(int_neg) (upper); sum dE_S(int_pos) (two-sided); sum J(val_pc1) and sum dE_S(val_pc1) (two-sided).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import holm, p_lower, p_upper

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
COLS = ("dE_S", "dE_C", "dslope_S", "dslope_C", "dE_S_sev_minus_mild", "dslope_S_sev_minus_mild")
NULL_FILE = {"pc1_neg": "pair_neg", "pc1_pos": "pair_pos", "int_neg": "int_neg", "int_pos": "int_pos",
             "val_pc1": "val_pc1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def jj(m):
        f = R / f"s1l_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, c, n):
        """(observed, null draws) of column c ('J' = dslope_S - dslope_C) for direction n in model m."""
        e = jj(m)["readout"]["R9k"]["effects"][n]
        z = np.load(R / f"s1l_null_{NULL_FILE[n]}_{m}_anger_d{a.depth}.npy")
        if c == "J":
            return e["dslope_S"] - e["dslope_C"], z[:, COLS.index("dslope_S")] - z[:, COLS.index("dslope_C")]
        return e[c], z[:, COLS.index(c)]

    gate, checks = [], {}
    for m in MODELS:
        j = jj(m)
        if not j or "t_total_s" not in j:
            continue
        r, mc = j["readout"]["R9k"], j["manipulation"]
        c = {k: j["checks"][k] for k in ("severity_effect", "severity_frac_pos", "C_severity_effect",
                                         "C_rho_location", "S_base_mild", "S_base_severe", "C_base_mean")}
        c |= {f"{k}_p": r["null"][k]["p_one"] for k in ("pos_82", "pos_sev")}
        c["valence_pos_minus_neg"] = mc["valence_pos_minus_neg"]["mean"]
        for hi, lo in (("anger_L", "anger_lo"), ("joy_L", "joy_lo")):
            x = mc[f"intensity_{hi}_minus_{lo}"]
            c[f"intensity_{hi}"] = x["mean"]
            c[f"intensity_{hi}_frac"] = x["frac_pos"]
            c[f"intensity_ok_{hi}"] = x["mean"] > 0.5 and x["frac_pos"] >= 0.75
        c["pair_cos_err"] = r["null"]["pair"]["cos_got_max_abs_err"]
        c["cos_with_1g_pc1_neg"] = j["cos_with_1g"]["pc1_neg"]
        c["pass"] = ((c["pos_82_p"] < 0.05 or c["pos_sev_p"] < 0.05) and c["severity_effect"] > 1
                     and c["severity_frac_pos"] >= 0.8 and c["C_rho_location"] > 0.5
                     and abs(c["C_severity_effect"]) < 1 and c["valence_pos_minus_neg"] > 2)
        checks[m] = c
        if c["pass"]:
            gate.append(m)
    g_neg = [m for m in gate if checks[m]["intensity_ok_anger_L"]]
    g_pos = [m for m in gate if checks[m]["intensity_ok_joy_L"]]

    def stat(models, terms):
        obs, nul = 0.0, 0.0
        for m in models:
            for sign, c, n in terms:
                x, z = src(m, c, n)
                obs, nul = obs + sign * x, nul + sign * z
        if not models:
            return {"models": [], "not_testable": True}
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": float(obs), "null_mean": float(np.mean(nul)),
                "null_sd": float(np.std(nul, ddof=1)), "null_q05": float(np.quantile(nul, 0.05)),
                "null_q95": float(np.quantile(nul, 0.95)), "p_upper": pu, "p_lower": pl,
                "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate": gate, "gate_int_neg": g_neg, "gate_int_pos": g_pos, "checks": checks}
    prim = {}
    if len(gate) >= 2:
        prim["L1_J_pc1neg_minus_pc1pos"] = stat(gate, [(1, "J", "pc1_neg"), (-1, "J", "pc1_pos")])
    if len(g_neg) >= 2:
        prim["L2_J_int_neg"] = stat(g_neg, [(1, "J", "int_neg")])
    adj = holm({k: v["p_upper"] for k, v in prim.items()}) if prim else {}
    for k in prim:
        prim[k]["p_holm"] = adj[k] * (2 / len(prim)) if len(prim) < 2 else adj[k]   # Holm over two registered tests
        prim[k]["p_holm"] = min(1.0, prim[k]["p_holm"])
        prim[k]["significant"] = prim[k]["p_holm"] < 0.05
    out["primary"] = prim or {"not_testable": True}
    if len(gate) >= 2:
        sec = {"L1e_dE_S_pc1neg_minus_pc1pos": stat(gate, [(1, "dE_S", "pc1_neg"), (-1, "dE_S", "pc1_pos")]),
               "L1s_dslope_S_pc1neg_minus_pc1pos": stat(gate, [(1, "dslope_S", "pc1_neg"), (-1, "dslope_S", "pc1_pos")]),
               "J_pc1_neg": stat(gate, [(1, "J", "pc1_neg")]),
               "J_pc1_pos": stat(gate, [(1, "J", "pc1_pos")]),
               "dE_S_pc1_pos": stat(gate, [(1, "dE_S", "pc1_pos")]),
               "J_int_pos": stat(g_pos, [(1, "J", "int_pos")]),
               "dE_S_int_neg": stat(g_neg, [(1, "dE_S", "int_neg")]),
               "dE_S_int_pos": stat(g_pos, [(1, "dE_S", "int_pos")]),
               "J_val_pc1": stat(gate, [(1, "J", "val_pc1")]),
               "dE_S_val_pc1": stat(gate, [(1, "dE_S", "val_pc1")])}
        out["secondary"] = sec
    out["descriptive_single"] = {m: {"effects": jj(m)["readout"]["R9k"]["effects"], "cos": jj(m)["cos"]}
                                 for m in checks}
    (R / "s1l_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "gate_int_neg", "gate_int_pos", "checks", "primary")}
                     | {"secondary": out.get("secondary")}, indent=1))


if __name__ == "__main__":
    main()
