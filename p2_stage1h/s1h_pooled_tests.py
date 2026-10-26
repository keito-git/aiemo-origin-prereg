"""Stage 1h registered tests (CPU only; reads s1h_crossed.py outputs, and s1g outputs for the B2 comparison).

Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model).
Gate (per model): pos_B2x p < 0.05; valence check mean_gap_B2x > 0 (lenient wording more negative);
judgment check corr_within_pair > 0. If fewer than two models pass, H1 is not testable.
Amendment 01: mean_gap_B2 < 0 was removed from the gate (reported only); H1 needs the crossing in B2x, not in B2.
Primary: H1 = sum over gated models of Delta_B2x(pc1_neg). Upper p < 0.05: the judgment account.
         Lower p < 0.05: the negative-token account. (Two one-sided readings of one test, reported as p_upper/p_lower.)
Secondary: the same sum for word_anger, word_eval, word_poseval (two-sided); and
           sum [Delta_B2(pc1_neg) - Delta_B2x(pc1_neg)] (the size of the wording effect, two-sided).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import p_lower, p_upper

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="Stage 1h results dir")
    ap.add_argument("--res1g", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R, G = Path(a.res), Path(a.res1g)

    def jh(m):
        f = R / f"s1h_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, r, n):
        if r == "B2x":
            return jh(m)["readout"]["B2x"]["delta"][n], np.load(R / f"s1h_null_B2x_{n}_{m}_anger_d{a.depth}.npy")
        j = json.load(open(G / f"s1g_{m}_anger_d{a.depth}.json"))
        return j["readout"]["B2"]["delta"][n], np.load(G / f"s1g_null_B2_{n}_{m}_anger_d{a.depth}.npy")

    gate, checks = [], {}
    for m in MODELS:
        j = jh(m)
        if not j or "null" not in j.get("readout", {}).get("B2x", {}):
            continue
        c = {"pos_p": j["readout"]["B2x"]["null"]["pos"]["p_one"],
             "mean_gap_B2x": j["valence_check"]["mean_gap_B2x"], "mean_gap_B2": j["valence_check"]["mean_gap_B2"],
             "corr_within_pair": j["judgment_check"]["corr_within_pair"], "corr_raw": j["judgment_check"]["corr_raw"],
             "cos_with_1g_min": min(j["cos_with_1g"].values())}
        c["pass"] = c["pos_p"] < 0.05 and c["mean_gap_B2x"] > 0 and c["corr_within_pair"] > 0
        checks[m] = c
        if c["pass"]:
            gate.append(m)

    def stat(models, terms):
        obs, nul = 0.0, 0.0
        for m in models:
            for sign, r, n in terms:
                x, z = src(m, r, n)
                obs, nul = obs + sign * x, nul + sign * z
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": obs, "null_mean": float(np.mean(nul)), "null_sd": float(np.std(nul, ddof=1)),
                "null_q05": float(np.quantile(nul, 0.05)), "null_q95": float(np.quantile(nul, 0.95)),
                "p_upper": pu, "p_lower": pl, "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate": gate, "checks": checks}
    if len(gate) >= 2:
        out["primary"] = {"H1_B2x_pc1_neg": stat(gate, [(1, "B2x", "pc1_neg")])}
        out["secondary"] = {f"B2x_{n}": stat(gate, [(1, "B2x", n)]) for n in ("word_anger", "word_eval", "word_poseval")}
        out["secondary"]["B2_minus_B2x_pc1_neg"] = stat(gate, [(1, "B2", "pc1_neg"), (-1, "B2x", "pc1_neg")])
    else:
        out["primary"] = {"H1_B2x_pc1_neg": {"models": gate, "not_testable": True}}
    out["descriptive_single"] = {m: {"delta": jh(m)["readout"]["B2x"]["delta"],
                                     "p_one": {n: v["p_one"] for n, v in jh(m)["readout"]["B2x"]["null"].items()},
                                     "p_one_lower": {n: v["p_one_lower"] for n, v in jh(m)["readout"]["B2x"]["null"].items()},
                                     "base_mean": jh(m)["readout"]["B2x"]["base_mean"]} for m in checks}
    (R / "s1h_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
