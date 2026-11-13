"""Stage 1s: readout validity of the post-training checkpoints, with the Stage 1o criteria unchanged
(s1o_pooled_tests.validity, gradient positive control against the pair_neg nulls on the even scenarios, model entry
rule). Reads s1o_rating.py outputs for any model tag (s1o_pooled_tests.py has the six Stage 1o models hard-coded).
CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1o_pooled_tests import CHO, DIG, ok, validity
from s1o_rating import COLS, CTRL, JUDG, READ


def check(R: Path, m: str, depth: str) -> dict:
    tag = f"{m}_anger_d{depth}"
    j = json.load(open(R / f"s1o_{tag}.json"))
    if "t_total_s" not in j:
        return {"complete": False}
    nul = np.load(R / f"s1o_null_pair_neg_{tag}.npy")
    s, v = validity(j["base"]["scores"])
    pos_p = {t: float((np.sum(nul[:, COLS.index(f"even_{t}")] >= j["pos"]["obs"][t]) + 1) / (len(nul) + 1))
             for t in READ}
    valid = {t: bool(v[t] and pos_p[t] < .05) for t in READ}
    mc = j["manipulation"]
    anger_ok = ok(mc["intensity_anger_L_minus_anger_lo"]) and ok(mc["extremity_anger_L_minus_anger_lo"])
    vj = [t for t in JUDG if valid[t]]
    enter = (len(vj) >= 3 and any(t in DIG for t in vj) and any(t in CHO for t in vj)
             and any(valid[t] for t in CTRL) and anger_ok)
    return {"complete": True, "n_null_pair": int(len(nul)), "min_attainable_p": 1.0 / (len(nul) + 1),
            "summ": s, "valid_noSteer": v, "pos_p": pos_p, "valid": valid, "valid_judgment": vj,
            "valid_control": [t for t in CTRL if valid[t]], "anger_ok": anger_ok, "enter": enter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="directory with s1o_rating.py outputs")
    ap.add_argument("--models", nargs="+", required=True, help="model basenames, e.g. Llama-3.1-Tulu-3-8B-SFT")
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {m: check(Path(a.res), m, a.depth) for m in a.models}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        if not r["complete"]:
            print(m, "incomplete")
            continue
        print(m, "enter", r["enter"], "judgment", r["valid_judgment"], "control", r["valid_control"],
              "anger_ok", r["anger_ok"])
        print("  noSteer", {t: r["valid_noSteer"][t] for t in READ})
        print("  pos_p", {t: round(r["pos_p"][t], 3) for t in READ})


if __name__ == "__main__":
    main()
