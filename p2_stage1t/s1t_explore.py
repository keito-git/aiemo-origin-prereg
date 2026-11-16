"""Stage 1t exploratory (P2-S1t_exploratory_plan.md): excess beyond alignment with the judgment readouts fixed to
S and HARSH for every checkpoint (base, SFT, DPO, final). The two base models failed the Stage 1o entry rule (registered
reading "not measurable"); this comparison does not change that reading and supports no confirmatory claim.
Definitions as s1s_excess.model_excess, with vj = ["S", "HARSH"] for all models (Stage 1s values are NOT reused).
Per family: paired differences over the 16 directions between consecutive stages (descriptive t, two-sided).
--check: recompute with each model's registered valid readouts and assert equality with the registered excess files
(run before registration; S+HARSH values are not computed in that mode). CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from s1p_rating import CONTRASTS
from s1s_excess import model_excess

VJ = ["S", "HARSH"]
FAMILY = {
    "Llama": ["Llama-3.1-8B", "Llama-3.1-Tulu-3-8B-SFT", "Llama-3.1-Tulu-3-8B-DPO", "Llama-3.1-Tulu-3-8B"],
    "OLMo": ["OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"],
    "Llama-Meta": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct"],
}


def keep(r: dict) -> dict:
    return {k: r[k] for k in ("valid_judg", "n_null", "mean", "sd", "n_pos", "t", "p_one_sided", "tost",
                              "mean_category", "mean_magnitude", "excess", "null_slope", "null_resid_sd")}


def paired(a: dict, b: dict) -> dict:
    d = np.array([b["excess"][c] - a["excess"][c] for c in CONTRASTS])
    t, p = stats.ttest_1samp(d, 0.0)
    return {"mean_diff_later_minus_earlier": float(d.mean()), "n_pos": int((d > 0).sum()),
            "t": float(t), "p_two_sided_descriptive": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="RESDIR=MODEL[,MODEL...] pairs: s1p_rating.py output dir and the models in it")
    ap.add_argument("--registered", nargs="*", default=[], help="registered s1s_excess-format files (for --check)")
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    where = {}
    for s in a.src:
        d, ms = s.split("=")
        where.update({m: Path(d) for m in ms.split(",")})
    if a.check:
        reg = {}
        for f in a.registered:
            reg.update({m: r for m, r in json.load(open(f)).items() if "mean" in r})
        out = {}
        for m, r in reg.items():
            x = model_excess(where[m], m, r["valid_judg"], a.depth)
            dmax = max(abs(x["excess"][c] - r["excess"][c]) for c in CONTRASTS)
            assert dmax < 1e-9, (m, dmax)
            out[m] = {"max_abs_diff": dmax}
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    res = {m: keep(model_excess(R, m, VJ, a.depth)) for m, R in where.items()}
    fam = {}
    for f, ms in FAMILY.items():
        ms = [m for m in ms if m in res]
        fam[f] = {"order": ms, "mean": [res[m]["mean"] for m in ms],
                  "steps": {f"{x} -> {y}": paired(res[x], res[y]) for x, y in zip(ms, ms[1:])}}
    out = {"note": "exploratory, S+HARSH only; base models 'not measurable' under the registered rule",
           "readouts": VJ, "models": res, "families": fam}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for m, r in res.items():
        print(f"{m:28s} n_null {r['n_null']:3d}  mean {r['mean']:+.2f}  {r['n_pos']}/16  t {r['t']:+.2f}")


if __name__ == "__main__":
    main()
