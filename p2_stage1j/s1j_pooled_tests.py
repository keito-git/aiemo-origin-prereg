"""Stage 1j registered tests (CPU only; reads s1j_rating.py outputs).

Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model).
Null arrays are (n, 2) = (Delta_S, Delta_O) of the same random direction, so S - O uses paired draws.
Gate (per model): pos p < 0.05; severity_effect > 1 (rating points) and severity_frac_pos >= 0.8 (the rating tracks
the mild/severe manipulation); object_rho_logweight > 0.5. If fewer than two models pass, the primary tests are not
testable.
Primary (Holm over two, one-sided upper):
  K1 = sum Delta_S(pc1_neg): the deserved-sanction rating rises (a stricter judgment, no option words).
  K2 = sum [Delta_S(pc1_neg) - Delta_O(pc1_neg)]: it rises more than an unrelated heaviness rating
       (excludes a general bias toward large/heavy ratings).
Secondary (no correction): sum Delta_O(pc1_neg) (two-sided; the heaviness bias itself);
  sum Delta_S for word_eval, word_poseval (upper), word_anger, val_eval, pc1_val (two-sided);
  sum [Delta_S(word_eval) - Delta_S(word_poseval)] (two-sided); sum [Delta_S - Delta_O] for word_eval, word_poseval,
  val_eval, pc1_val (two-sided).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import holm, p_lower, p_upper

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")
TASK = {"S": 0, "O": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def jj(m):
        f = R / f"s1j_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, t, n):
        return jj(m)["readout"]["R9"][f"delta_{t}"][n], np.load(R / f"s1j_null_{n}_{m}_anger_d{a.depth}.npy")[:, TASK[t]]

    gate, checks = [], {}
    for m in MODELS:
        j = jj(m)
        if not j or "pos" not in j.get("readout", {}).get("R9", {}).get("null", {}):
            continue
        c = {k: j["checks"][k] for k in ("severity_effect", "severity_frac_pos", "object_rho_logweight",
                                         "judgment_rho_B2_within_pair")}
        c |= {"pos_p": j["readout"]["R9"]["null"]["pos"]["p_one"], "pos_delta": j["readout"]["R9"]["pos_even"]["delta"],
              "cos_with_1g_min": min(j["cos_with_1g"].values())}
        c["pass"] = (c["pos_p"] < 0.05 and c["severity_effect"] > 1 and c["severity_frac_pos"] >= 0.8
                     and c["object_rho_logweight"] > 0.5)
        checks[m] = c
        if c["pass"]:
            gate.append(m)

    def stat(models, terms):
        obs, nul = 0.0, 0.0
        for m in models:
            for sign, t, n in terms:
                x, z = src(m, t, n)
                obs, nul = obs + sign * x, nul + sign * z
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": obs, "null_mean": float(np.mean(nul)), "null_sd": float(np.std(nul, ddof=1)),
                "null_q05": float(np.quantile(nul, 0.05)), "null_q95": float(np.quantile(nul, 0.95)),
                "p_upper": pu, "p_lower": pl, "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate": gate, "checks": checks}
    if len(gate) >= 2:
        prim = {"K1_S_pc1_neg": stat(gate, [(1, "S", "pc1_neg")]),
                "K2_S_minus_O_pc1_neg": stat(gate, [(1, "S", "pc1_neg"), (-1, "O", "pc1_neg")])}
        adj = holm({k: v["p_upper"] for k, v in prim.items()})
        for k in prim:
            prim[k]["p_holm"] = adj[k]
            prim[k]["significant"] = adj[k] < 0.05
        out["primary"] = prim
        sec = {"O_pc1_neg": stat(gate, [(1, "O", "pc1_neg")])}
        sec |= {f"S_{n}": stat(gate, [(1, "S", n)])
                for n in ("word_eval", "word_poseval", "word_anger", "val_eval", "pc1_val")}
        sec["S_eval_minus_poseval"] = stat(gate, [(1, "S", "word_eval"), (-1, "S", "word_poseval")])
        sec |= {f"S_minus_O_{n}": stat(gate, [(1, "S", n), (-1, "O", n)])
                for n in ("word_eval", "word_poseval", "val_eval", "pc1_val")}
        out["secondary"] = sec
    else:
        out["primary"] = {"not_testable": True, "models": gate}
    out["descriptive_single"] = {m: {"delta_S": jj(m)["readout"]["R9"]["delta_S"],
                                     "delta_O": jj(m)["readout"]["R9"]["delta_O"],
                                     "base_mean": jj(m)["readout"]["R9"]["base_mean"]} for m in checks}
    (R / "s1j_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
