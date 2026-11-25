"""H_VA^Sun-matched (prereg v3.1 §10): the Sun pipeline with k chosen to match Sun's reported recovery, CPU only.

Why: the registered H_VA^Sun chooses (k, lambda) by leave-one-label-out CV; with it, arousal reaches in-sample r = .65
at best (Sun reports .87 and states neither k nor lambda). In-sample arousal recovery rises with k on 27 labels
(k = 15 -> ~.80, k = 26 -> ~.9-1.0), so Sun's value is reproducible only with many components. This variant is added
as a FOURTH human candidate for K1 (it does not enter A), so that T1 is not judged against a weaker Sun axis.

Rule (fixed before any task gradient): at each layer, the smallest k in 2..26 (lambda = 0, V first in Gram-Schmidt,
same steering vectors and self-report targets as H_VA^Sun) with in-sample r_V >= 0.90 and r_A >= 0.80 after
Gram-Schmidt. The LOLO CV r of that k is reported alongside (it shows how much of the in-sample value is overfit).
Output: <out> (part file with no directions and the subspace H_VA_sun_matched at --layer) and <out>.report.json.
"""
import argparse
import json

import numpy as np

import p3k_directions as d

K_GRID = range(2, 27)
R_V_MIN, R_A_MIN = 0.90, 0.80


def fit_k(V: np.ndarray, yv: np.ndarray, ya: np.ndarray, k: int) -> dict:
    wv, _ = d.sun_fit(V, yv, k, 0.0)
    wa, _ = d.sun_fit(V, ya, k, 0.0)
    v, a, c = d.gram_schmidt_va(wv, wa)
    Vc = V - V.mean(0)
    return {"v": v, "a": a, "r_V": float(np.corrcoef(Vc @ v, yv)[0, 1]),
            "r_A": float(np.corrcoef(Vc @ a, ya)[0, 1]), "cos_before_GS": c}


def lolo_r(V: np.ndarray, y: np.ndarray, k: int) -> float:
    cv = d.sun_lolo(V, y, ks=(k,), lams=np.array([0.0]))
    return cv["cv_r"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True, help="<sun-va out>.vectors.npz")
    p.add_argument("--ratings", required=True, help="sun_self_report.json")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    z = np.load(a.vectors)
    layers = [int(x) for x in z["layers"]]
    yv, ya = d._ratings_from_report(json.load(open(a.ratings, encoding="utf-8")))
    assert not (np.isnan(yv).any() or np.isnan(ya).any()), "unparsed self-report labels: rule not defined"
    rep = {"rule": f"smallest k in 2..26, lambda=0, in-sample r_V >= {R_V_MIN} and r_A >= {R_A_MIN} after GS",
           "per_layer": {}}
    basis = None
    for li, L in enumerate(layers):
        V = z["V"][li].astype(np.float64)
        hit = None
        for k in K_GRID:
            f = fit_k(V, yv, ya, k)
            if f["r_V"] >= R_V_MIN and f["r_A"] >= R_A_MIN:
                hit = (k, f)
                break
        if hit is None:
            rep["per_layer"][str(L)] = {"k": None}
        else:
            k, f = hit
            rep["per_layer"][str(L)] = {"k": k, "r_V": f["r_V"], "r_A": f["r_A"],
                                        "cos_before_GS": f["cos_before_GS"],
                                        "lolo_r_V": lolo_r(V, yv, k), "lolo_r_A": lolo_r(V, ya, k)}
            if L == a.layer:
                basis = d.orth_basis(np.stack([f["v"], f["a"]], 1))
        print(L, rep["per_layer"][str(L)], flush=True)
    rep["layer"] = a.layer
    rep["chosen"] = rep["per_layer"].get(str(a.layer))
    stem = a.out[:-4] if a.out.endswith(".npz") else a.out
    open(stem + ".report.json", "w").write(json.dumps(rep, indent=1))
    assert basis is not None, f"no k reaches the Sun-matched recovery at layer {a.layer}"
    d.save_part(a.out, np.zeros((0, basis.shape[0])), [], [],
                {"cmd": "sun-matched", "layer": a.layer, "rule": rep["rule"], "fit": rep["chosen"]},
                H_VA_sun_matched=basis)


if __name__ == "__main__":
    main()
