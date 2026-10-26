"""Stage 1i registered tests (CPU only; reads s1i_neutral.py outputs).

Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model).
Gate (per model): pos_B2n p < 0.05; neutral wording |mean_gap_B2n| < 1 (B2x was +3.9 to +6.6); judgment check
corr_within_pair > 0. If fewer than two models pass, the tests are not testable.
Primary (Holm over two):
  I1 = sum Delta_B2n(pc1_neg), one-sided upper (a wording-independent stricter judgment)
  I2 = sum [Delta_B2n(word_eval) - Delta_B2n(word_poseval)], two-sided (does negative evaluation still exceed
       positive evaluation once wording valence is neutral?)
Secondary (no correction): sum Delta_B2n of word_eval, word_poseval (upper), word_anger (two-sided);
each also divided per model by that model's positive-control Delta_B2n (sensitivity-normalised, descriptive).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import holm, p_lower, p_upper

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def ji(m):
        f = R / f"s1i_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, n):
        return ji(m)["readout"]["B2n"]["delta"][n], np.load(R / f"s1i_null_B2n_{n}_{m}_anger_d{a.depth}.npy")

    gate, checks = [], {}
    for m in MODELS:
        j = ji(m)
        if not j or "null" not in j.get("readout", {}).get("B2n", {}):
            continue
        c = {"pos_p": j["readout"]["B2n"]["null"]["pos"]["p_one"], "pos_delta": j["readout"]["B2n"]["pos_even"]["delta"],
             "mean_gap_B2n": j["valence_check"]["mean_gap_B2n"], "mean_gap_B2": j["valence_check"]["mean_gap_B2"],
             "corr_within_pair": j["judgment_check"]["corr_within_pair"], "cos_with_1g_min": min(j["cos_with_1g"].values())}
        c["pass"] = c["pos_p"] < 0.05 and abs(c["mean_gap_B2n"]) < 1 and c["corr_within_pair"] > 0
        checks[m] = c
        if c["pass"]:
            gate.append(m)

    def stat(models, terms, norm=False):
        obs, nul = 0.0, 0.0
        for m in models:
            k = 1.0 / checks[m]["pos_delta"] if norm else 1.0
            for sign, n in terms:
                x, z = src(m, n)
                obs, nul = obs + sign * k * x, nul + sign * k * z
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": obs, "null_mean": float(np.mean(nul)), "null_sd": float(np.std(nul, ddof=1)),
                "null_q05": float(np.quantile(nul, 0.05)), "null_q95": float(np.quantile(nul, 0.95)),
                "p_upper": pu, "p_lower": pl, "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate": gate, "checks": checks}
    if len(gate) >= 2:
        prim = {"I1_B2n_pc1_neg": stat(gate, [(1, "pc1_neg")]),
                "I2_B2n_eval_minus_poseval": stat(gate, [(1, "word_eval"), (-1, "word_poseval")])}
        adj = holm({"I1_B2n_pc1_neg": prim["I1_B2n_pc1_neg"]["p_upper"],
                    "I2_B2n_eval_minus_poseval": prim["I2_B2n_eval_minus_poseval"]["p_two"]})
        for k in prim:
            prim[k]["p_holm"] = adj[k]
            prim[k]["significant"] = adj[k] < 0.05
        out["primary"] = prim
        out["secondary"] = {f"B2n_{n}": stat(gate, [(1, n)]) for n in ("word_eval", "word_poseval", "word_anger")}
        out["secondary"] |= {f"B2n_{n}_pos_normalised": stat(gate, [(1, n)], norm=True)
                             for n in ("pc1_neg", "word_eval", "word_poseval", "word_anger")}
    else:
        out["primary"] = {"not_testable": True, "models": gate}
    out["descriptive_single"] = {m: {"delta": ji(m)["readout"]["B2n"]["delta"],
                                     "p_one": {n: v["p_one"] for n, v in ji(m)["readout"]["B2n"]["null"].items()},
                                     "p_one_lower": {n: v["p_one_lower"] for n, v in ji(m)["readout"]["B2n"]["null"].items()},
                                     "base_mean": ji(m)["readout"]["B2n"]["base_mean"]} for m in checks}
    (R / "s1i_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
