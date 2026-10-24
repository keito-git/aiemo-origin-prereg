"""Stage 1f registered tests, summed over the three models (CPU only; reads s1f_valence_rarity.py outputs).

Null of a sum/difference = the same sum/difference of the saved null draws, index i with index i.
Gate: the Stage 1d positive control for B2 (anger situations) must have p < 0.05, else the model is dropped.
Primary (Holm over the two):
  F_val  = sum_m [Delta(word_eval) - Delta(word_poseval)], one-sided (upper)
  F_rare = sum_m Delta(word_rare), one-sided (upper)
Secondary: F_pos = sum_m Delta(word_poseval), two-sided; F_sit = sum_m Delta(sit_scaled), two-sided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1d_pooled_tests import MODELS, holm, p_lower, p_upper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--res1d", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R, R1 = Path(a.res), Path(a.res1d)

    def j1f(m):
        f = R / f"s1f_{m}_anger_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def null(m, n):
        return np.load(R / f"s1f_null_B2_{n}_{m}_anger_d{a.depth}.npy")

    ok = []
    for m in MODELS:
        f = R1 / f"s1d_{m}_anger_d{a.depth}.json"
        if j1f(m) and f.exists() and json.load(open(f))["readout"]["B2"]["null"]["pos"]["p_one"] < 0.05:
            ok.append(m)

    def contrast(p, q=None):
        obs, nul = 0.0, 0.0
        for m in ok:
            d = j1f(m)["B2"]["delta"]
            obs += d[p] - (d[q] if q else 0.0)
            nul = nul + null(m, p) - (null(m, q) if q else 0.0)
        pu, pl = p_upper(nul, obs), p_lower(nul, obs)
        return {"models": ok, "obs": obs, "null_mean": float(np.mean(nul)), "null_q95": float(np.quantile(nul, 0.95)),
                "p_upper": pu, "p_lower": pl, "p_two": min(1.0, 2 * min(pu, pl))}

    out = {"gate_B2": ok}
    prim = {"F_val_eval_minus_poseval": contrast("word_eval", "word_poseval"), "F_rare": contrast("word_rare")}
    adj = holm({k: v["p_upper"] for k, v in prim.items()})
    for k in prim:
        prim[k]["p_holm"] = adj[k]
        prim[k]["significant"] = adj[k] < 0.05
    out["primary"] = prim
    out["secondary"] = {"F_pos_poseval": contrast("word_poseval"), "F_sit_scaled": contrast("sit_scaled")}
    out["descriptive_single"] = {m: {"delta": j1f(m)["B2"]["delta"], "p_one": {n: v["p_one"] for n, v in j1f(m)["B2"]["null"].items()},
                                     "closing_surprisal": j1f(m)["closing_surprisal"], "cos_with_1d": j1f(m).get("cos_with_1d")}
                                 for m in MODELS if j1f(m)}
    (R / "s1f_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
