"""Stage 1s registered tests (CPU only; reads s1s_excess.py output). Design 02_設計/15_段階1s_..._draft_v2.
H-stage (primary): Tulu-3-8B SFT and DPO; one-sided t of mean excess > 0 (df 15), Holm over the two.
  "positive" = Holm-adjusted p < .05. TOST (+-0.5) per checkpoint, secondary; "zero" only if TOST p < .05.
H-recipe (secondary, two separate tests):
  OLMo axis: OLMo-2-1124-7B SFT and DPO, one-sided t, Holm over the two.
  Llama axis: Llama-3.1-8B-Instruct, one-sided t.
A checkpoint whose readouts fail the Stage 1o validity rule is "not measurable" and is dropped from its Holm family.
Reading table for H-stage (fixed before the run): see the preregistration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGE = ("Llama-3.1-Tulu-3-8B-SFT", "Llama-3.1-Tulu-3-8B-DPO")
OLMO = ("OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO")
LLAMA = ("Llama-3.1-8B-Instruct",)


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = run
    return out


def family(E: dict, models: tuple) -> dict:
    meas = [m for m in models if m in E and "mean" in E[m]]
    adj = holm({m: E[m]["p_one_sided"] for m in meas}) if meas else {}
    res = {}
    for m in models:
        if m not in E:
            res[m] = {"status": "missing"}
        elif "mean" not in E[m]:
            res[m] = {"status": "not measurable"}
        else:
            r = E[m]
            pos = adj[m] < .05
            zero = (not pos) and r["tost"]["p"] < .05
            res[m] = {"mean": r["mean"], "sd": r["sd"], "n_pos": r["n_pos"], "t": r["t"],
                      "p_one_sided": r["p_one_sided"], "p_holm": adj[m], "tost_p": r["tost"]["p"],
                      "status": "positive" if pos else ("zero (TOST)" if zero else "not positive, undecided"),
                      "mean_category": r["mean_category"], "mean_magnitude": r["mean_magnitude"]}
    return res


def reading(st: dict) -> str:
    s = [st[m]["status"] for m in STAGE]
    if any(x in ("missing", "not measurable") for x in s):
        return "stage axis not fully measurable: " + ", ".join(f"{m}={x}" for m, x in zip(STAGE, s))
    p = [x == "positive" for x in s]
    if p[0] and p[1]:
        return "row 1: residual arises at SFT and survives DPO"
    if p[0]:
        return "row 2: residual at SFT, not positive at DPO, present in the final model (non-monotonic)"
    if p[1]:
        return "row 3: residual arises at DPO"
    if all(x == "zero (TOST)" for x in s):
        return "row 4 (strong): residual arises at the final RLVR stage (both earlier stages equivalent to zero)"
    return "row 4: suggests the residual arises at the final RLVR stage (earlier stages not positive, not shown zero)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excess", required=True, help="s1s_excess.py output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = json.load(open(a.excess))
    st = family(E, STAGE)
    out = {"H-stage": st, "H-stage_reading": reading(st),
           "H-recipe_OLMo": family(E, OLMO), "H-recipe_Llama": family(E, LLAMA)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
