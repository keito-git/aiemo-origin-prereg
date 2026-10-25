"""Stage 1g registered tests (CPU only; reads s1g_confirm.py outputs, and s1f outputs for the Stage 1c-1f models).

Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i (within a model).
Gates: B2 tests use models with pos_B2 p < 0.05. B2neg tests use models with pos_B2neg p < 0.05 AND
corr(B2 - B2neu, B2neg + B2neu) > 0 over items (the negated frame is understood; Amendment 01 replaced the raw
corr(B2, B2neg), which the shared option prior drives negative). If fewer than two models pass, G3 is not testable
and enters Holm with p = 1.
Primary (Holm over three):
  G1 = sum_new [Delta_B2(word_anger) - Delta_B2(sit)], one-sided upper   (replicates Stage 1d primary 1)
  G2 = sum_new [Delta_B2(word_eval) - Delta_B2(word_poseval)], upper    (replicates Stage 1f F_val)
  G3 = sum_all6 Delta_B2neg(pc1_neg), one-sided upper                    (judgment, not sanction-token bias)
Secondary (no correction): G2 over all six models (old models from Stage 1f); sum_all6 Delta_B2(pc1_neg) and
Delta_B2(anger_res), upper; sum_new Delta_B2(word_rare), upper; sum_all6 Delta_B2neg of word_anger / word_eval /
word_poseval, two-sided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import holm, p_lower, p_upper

OLD = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct")
NEW = ("Mistral-7B-Instruct-v0.3", "granite-3.1-8b-instruct", "Llama-3.1-Tulu-3-8B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="Stage 1g results dir")
    ap.add_argument("--res1f", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R, R1f = Path(a.res), Path(a.res1f)

    def jg(m):
        f = R / f"s1g_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def src(m, r, n):
        """(observed Delta, null draws) for model m, readout r, direction n; s1g first, s1f for old-model B2."""
        j = jg(m)
        if j and n in j.get("readout", {}).get(r, {}).get("delta", {}):
            return j["readout"][r]["delta"][n], np.load(R / f"s1g_null_{r}_{n}_{m}_anger_d{a.depth}.npy")
        jf = R1f / f"s1f_{m}_anger_d{a.depth}.json"
        if r == "B2" and jf.exists():
            d = json.load(open(jf))["B2"]["delta"]
            if n in d:
                return d[n], np.load(R1f / f"s1f_null_B2_{n}_{m}_anger_d{a.depth}.npy")
        raise KeyError((m, r, n))

    def gate(models, r):
        ok = []
        for m in models:
            j = jg(m)
            if not j or r not in j.get("readout", {}):
                continue
            g = j["readout"][r]["null"]["pos"]["p_one"] < 0.05
            if r == "B2neg":
                g = g and j["b2neg_check"]["corr_adj"] > 0
            if g:
                ok.append(m)
        return ok

    def stat(models, r, p, q=None):
        obs, nul = 0.0, 0.0
        for m in models:
            xp, zp = src(m, r, p)
            obs, nul = obs + xp, nul + zp
            if q:
                xq, zq = src(m, r, q)
                obs, nul = obs - xq, nul - zq
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": list(models), "obs": obs, "null_mean": float(np.mean(nul)), "null_sd": float(np.std(nul, ddof=1)),
                "null_q95": float(np.quantile(nul, 0.95)), "p_upper": pu, "p_lower": pl, "p_two": min(1.0, 2 * min(pu, pl))}

    g_new, g_all = gate(NEW, "B2"), gate(OLD + NEW, "B2")
    g_neg = gate(OLD + NEW, "B2neg")
    out = {"gate": {"B2_new": g_new, "B2_all": g_all, "B2neg_all": g_neg},
           "b2neg_check": {m: (jg(m) or {}).get("b2neg_check") for m in OLD + NEW}}
    prim = {"G1_new_anger_minus_sit": stat(g_new, "B2", "word_anger", "sit"),
            "G2_new_eval_minus_poseval": stat(g_new, "B2", "word_eval", "word_poseval"),
            "G3_all_B2neg_pc1_neg": stat(g_neg, "B2neg", "pc1_neg") if len(g_neg) >= 2 else
            {"models": g_neg, "not_testable": True, "p_upper": 1.0}}
    adj = holm({k: v["p_upper"] for k, v in prim.items()})
    for k in prim:
        prim[k]["p_holm"] = adj[k]
        prim[k]["significant"] = adj[k] < 0.05
    out["primary"] = prim
    out["secondary"] = {"G2_all_eval_minus_poseval": stat(g_all, "B2", "word_eval", "word_poseval"),
                        "all_B2_pc1_neg": stat(g_all, "B2", "pc1_neg"), "all_B2_anger_res": stat(g_all, "B2", "anger_res"),
                        "new_B2_word_rare": stat(g_new, "B2", "word_rare")}
    if g_neg:
        out["secondary"] |= {f"all_B2neg_{n}": stat(g_neg, "B2neg", n) for n in ("word_anger", "word_eval", "word_poseval")}
    out["descriptive_single"] = {m: {r: {"delta": v["delta"], "p_one": {n: x["p_one"] for n, x in v["null"].items()},
                                         "base_mean": v["base_mean"]} for r, v in jg(m)["readout"].items()}
                                 | {"pc1_share": jg(m)["pc1_share"]} for m in OLD + NEW if jg(m)}
    (R / "s1g_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
