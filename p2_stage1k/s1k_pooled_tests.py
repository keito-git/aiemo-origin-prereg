"""Stage 1k registered tests (CPU only; reads s1k_rating.py outputs).

Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model);
the columns of one null array come from the same random direction, so S - C contrasts use paired draws.
Gate (per model): positive control p < 0.05 for pos_82 or pos_sev; severity_effect > 1 and severity_frac_pos >= 0.8;
C_rho_location > 0.5 and |C_severity_effect| < 1. If fewer than two models pass, the primary tests are not testable.
Primary (Holm over two, one-sided upper):
  K1k = sum dE_S(pc1_neg): the sanction rating rises on new scenarios (replication of Stage 1j K1).
  K3  = sum [dslope_S - dslope_C](pc1_neg): it rises more than a location rating of the same text, on the
        saturation-free slope metric.
Secondary (no correction): K3e = sum [dE_S - dE_C](pc1_neg) (upper); sum dslope_S(pc1_neg) (upper);
  sum dE_C(pc1_neg), sum dslope_C(pc1_neg) (two-sided); severity interaction sum dslope_S_sev_minus_mild(pc1_neg)
  and sum dE_S_sev_minus_mild(pc1_neg) (two-sided); sum dE_S for word_eval, word_anger, val_eval (two-sided) and
  word_poseval (upper); sum [dslope_S - dslope_C] for word_eval, word_anger, val_eval, word_poseval (two-sided);
  sum [dE_S(word_eval) - dE_S(word_poseval)] (two-sided).
Descriptive: per model effects; the pos_82 direction ("say a larger digit") on S and C as a benchmark of a general
digit bias.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def jj(m):
        f = R / f"s1k_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, c, n):
        return (jj(m)["readout"]["R9k"]["effects"][n][c],
                np.load(R / f"s1k_null_{n}_{m}_anger_d{a.depth}.npy")[:, COLS.index(c)])

    gate, checks = [], {}
    for m in MODELS:
        j = jj(m)
        if not j or not {"pos_82", "pos_sev"} <= set(j.get("readout", {}).get("R9k", {}).get("null", {})):
            continue
        r = j["readout"]["R9k"]
        c = {k: j["checks"][k] for k in ("severity_effect", "severity_frac_pos", "C_severity_effect",
                                         "C_rho_location", "S_base_mild", "S_base_severe", "C_base_mean")}
        c |= {f"{k}_p": r["null"][k]["p_one"] for k in ("pos_82", "pos_sev")}
        c |= {f"{k}_delta": r["pos_even"][k] for k in ("pos_82", "pos_sev")}
        c["cos_with_1g_min"] = min(j["cos_with_1g"].values())
        c["pass"] = ((c["pos_82_p"] < 0.05 or c["pos_sev_p"] < 0.05) and c["severity_effect"] > 1
                     and c["severity_frac_pos"] >= 0.8 and c["C_rho_location"] > 0.5
                     and abs(c["C_severity_effect"]) < 1)
        checks[m] = c
        if c["pass"]:
            gate.append(m)

    def stat(models, terms):
        obs, nul = 0.0, 0.0
        for m in models:
            for sign, c, n in terms:
                x, z = src(m, c, n)
                obs, nul = obs + sign * x, nul + sign * z
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": obs, "null_mean": float(np.mean(nul)), "null_sd": float(np.std(nul, ddof=1)),
                "null_q05": float(np.quantile(nul, 0.05)), "null_q95": float(np.quantile(nul, 0.95)),
                "p_upper": pu, "p_lower": pl, "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate": gate, "checks": checks}
    if len(gate) >= 2:
        g = gate
        prim = {"K1k_dE_S_pc1_neg": stat(g, [(1, "dE_S", "pc1_neg")]),
                "K3_dslope_S_minus_C_pc1_neg": stat(g, [(1, "dslope_S", "pc1_neg"), (-1, "dslope_C", "pc1_neg")])}
        adj = holm({k: v["p_upper"] for k, v in prim.items()})
        for k in prim:
            prim[k]["p_holm"] = adj[k]
            prim[k]["significant"] = adj[k] < 0.05
        out["primary"] = prim
        sec = {"K3e_dE_S_minus_C_pc1_neg": stat(g, [(1, "dE_S", "pc1_neg"), (-1, "dE_C", "pc1_neg")]),
               "dslope_S_pc1_neg": stat(g, [(1, "dslope_S", "pc1_neg")]),
               "dE_C_pc1_neg": stat(g, [(1, "dE_C", "pc1_neg")]),
               "dslope_C_pc1_neg": stat(g, [(1, "dslope_C", "pc1_neg")]),
               "interaction_dslope_pc1_neg": stat(g, [(1, "dslope_S_sev_minus_mild", "pc1_neg")]),
               "interaction_dE_pc1_neg": stat(g, [(1, "dE_S_sev_minus_mild", "pc1_neg")])}
        for n in ("word_eval", "word_anger", "val_eval", "word_poseval"):
            sec[f"dE_S_{n}"] = stat(g, [(1, "dE_S", n)])
            sec[f"dslope_S_minus_C_{n}"] = stat(g, [(1, "dslope_S", n), (-1, "dslope_C", n)])
        sec["dE_S_eval_minus_poseval"] = stat(g, [(1, "dE_S", "word_eval"), (-1, "dE_S", "word_poseval")])
        out["secondary"] = sec
    else:
        out["primary"] = {"not_testable": True, "models": gate}
    out["descriptive_single"] = {m: {"effects": jj(m)["readout"]["R9k"]["effects"],
                                     "pos_effects_all_items": jj(m)["readout"]["R9k"]["pos_effects_all_items"]}
                                 for m in checks}
    (R / "s1k_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "checks", "primary")} | {"secondary": out.get("secondary")}, indent=1))


if __name__ == "__main__":
    main()
