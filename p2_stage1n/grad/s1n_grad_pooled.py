"""Pooled tests of the Stage 1n gradient decomposition (plan: 02_設計/10_勾配分解解析_登録案).
G0 (gate, per model): r between Dhat and the measured D_S,U_bright over the Stage 1n pair_neg nulls; r >= .5 enters
  G1/G2 (sensitivity: .3, .7).
G1 (main, one-sided): sum over gated models of z_res(int_neg), z standardized by the model's pair_neg null of A_res;
  null = the same sum over the null index j.
G2 (two-sided): sum of z_res(int_neg) - z_res(int_neu) (int_neu standardized by pair_neu); null = sum of
  z(r_neg_j) - z(r_neu_j).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1n_rating import COLS

I_S, I_U = COLS.index("dslope_S"), COLS.index("dslope_U_bright")


def zs(x, ref):
    return (x - ref.mean()) / ref.std(ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad", required=True)
    ap.add_argument("--res1n", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", type=float, default=0.5)
    a = ap.parse_args()
    G, R = Path(a.grad), Path(a.res1n)
    per = {}
    for f in sorted(G.glob("s1n_grad_*_anger_d0.33.json")):
        tag = f.name[len("s1n_grad_"):-len(".json")]
        j = json.load(open(f))
        keys = j["keys"]
        nn = {nm: np.load(G / f"s1n_grad_null_{nm}_{tag}.npy") for nm in ("pair_neg", "pair_neu")}
        meas = np.load(R / f"s1n_null_pair_neg_{tag}.npy")
        dmeas = meas[:, I_S] - meas[:, I_U]
        dhat = nn["pair_neg"][:, keys.index("Dhat")]
        r = float(np.corrcoef(dhat, dmeas)[0, 1])
        ent = {"r_G0": r, "rebuild_check": j.get("rebuild_check")}
        for key in ("main", "sens"):
            k = keys.index(f"A_res_{key}")
            ref_neg, ref_neu = nn["pair_neg"][:, k], nn["pair_neu"][:, k]
            ent[key] = {"z_neg": float(zs(j["dirs"]["int_neg"][f"A_res_{key}"], ref_neg)),
                        "z_neu": float(zs(j["dirs"]["int_neu"][f"A_res_{key}"], ref_neu)),
                        "null_neg": zs(ref_neg, ref_neg), "null_neu": zs(ref_neu, ref_neu)}
        per[tag] = ent

    res = {"per_model": {t: {"r_G0": e["r_G0"], "rebuild_check": e["rebuild_check"],
                             **{k: {"z_neg": e[k]["z_neg"], "z_neu": e[k]["z_neu"]} for k in ("main", "sens")}}
                         for t, e in per.items()}}
    for gate in sorted({a.gate, .3, .7}):
        use = [t for t, e in per.items() if e["r_G0"] >= gate]
        g = {"models": use}
        for key in ("main", "sens"):
            if not use:
                g[key] = None
                continue
            T1 = sum(per[t][key]["z_neg"] for t in use)
            N1 = sum(per[t][key]["null_neg"] for t in use)
            T2 = sum(per[t][key]["z_neg"] - per[t][key]["z_neu"] for t in use)
            N2 = sum(per[t][key]["null_neg"] - per[t][key]["null_neu"] for t in use)
            k1 = 2.0 if len(use) == 1 else 1.0   # one gated model only: p doubled (registered)
            g[key] = {"G1_T": float(T1),
                      "G1_p_one": min(1.0, k1 * float((np.sum(N1 >= T1) + 1) / (len(N1) + 1))),
                      "G2_T": float(T2),
                      "G2_p_two": min(1.0, k1 * float((np.sum(np.abs(N2) >= abs(T2)) + 1) / (len(N2) + 1)))}
        res[f"gate_{gate}"] = g
    res["registered"] = f"gate_{a.gate}/main"
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
