"""Stage 1d preregistered tests, pooled over models (CPU only; reads the outputs of s1d_word_specificity.py).

Null for a sum/difference of observed Deltas = the same sum/difference of the saved null draws, index i with index i
(each null draw matches the variance of its own direction). One-sided p = (#null >= obs + 1) / (n + 1);
two-sided p = min(1, 2 * min(p_upper, p_lower)).
Models whose positive control fails (p >= 0.05) on a readout are dropped from the sum for that readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ("Qwen3-8B", "llm-jp-4-8b-instruct", "OLMo-2-1124-7B-Instruct")


def p_upper(null, obs):
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def p_lower(null, obs):
    return float((np.sum(null <= obs) + 1) / (len(null) + 1))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - j) * ps[k]))
        adj[k] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--depth", default="0.33")
    a = ap.parse_args()
    R = Path(a.res)

    def load(model, sit):
        f = R / f"s1d_{model}_{sit}_d{a.depth}.json"
        return json.load(open(f)) if f.exists() else None

    def null(model, sit, readout, name):
        return np.load(R / f"s1d_null_{readout}_{name}_{model}_{sit}_d{a.depth}.npy")

    def models_ok(sit, readout):
        ok = []
        for m in MODELS:
            j = load(m, sit)
            if j and readout in j.get("readout", {}) and j["readout"][readout]["null"]["pos"]["p_one"] < 0.05:
                ok.append(m)
        return ok

    def contrast(sit, readout, p, q=None):
        ms = models_ok(sit, readout)
        obs, nul = 0.0, 0.0
        for m in ms:
            d = load(m, sit)["readout"][readout]["delta"]
            obs += d[p] - (d[q] if q else 0.0)
            nul = nul + null(m, sit, readout, p) - (null(m, sit, readout, q) if q else 0.0)
        return {"models": ms, "obs": obs, "null_mean": float(np.mean(nul)), "null_q95": float(np.quantile(nul, 0.95)),
                "p_upper": p_upper(nul, obs), "p_lower": p_lower(nul, obs),
                "p_two": min(1.0, 2 * min(p_upper(nul, obs), p_lower(nul, obs)))}

    out = {"gate": {f"{s}/{r}": models_ok(s, r) for s, r in (("anger", "B2"), ("anger", "B3"), ("fear", "B2"), ("fear", "B3"))}}
    out["primary1_word_anger_minus_sit"] = contrast("anger", "B2", "word_anger", "sit")
    out["primary1_supported"] = out["primary1_word_anger_minus_sit"]["p_upper"] < 0.05
    p2 = {x: contrast("anger", "B2", "word_anger", f"word_{x}") for x in ("eval", "fear", "joy")}
    adj = holm({x: v["p_upper"] for x, v in p2.items()})
    for x in p2:
        p2[x]["p_holm"] = adj[x]
        p2[x]["significant"] = adj[x] < 0.05
    out["primary2_word_anger_minus_word_X"] = p2
    out["secondary_crossing"] = {
        "fear/B2/word_fear": contrast("fear", "B2", "word_fear"), "fear/B2/sit": contrast("fear", "B2", "sit"),
        "anger/B3/word_anger": contrast("anger", "B3", "word_anger"), "anger/B3/sit": contrast("anger", "B3", "sit")}
    out["descriptive_single"] = {}
    for m in MODELS:
        for s in ("anger", "fear"):
            j = load(m, s)
            if j:
                out["descriptive_single"][f"{m}/{s}"] = {r: {n: {"delta": v["delta"].get(n), "p_one": v["null"][n]["p_one"]}
                                                             for n in v["null"] if n != "pos"} | {"pos_p": v["null"]["pos"]["p_one"]}
                                                         for r, v in j["readout"].items()}
    (R / "s1d_pooled_tests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
