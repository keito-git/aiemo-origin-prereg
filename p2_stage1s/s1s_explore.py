"""Stage 1s exploratory analyses (listed as exploratory in the preregistration; NOT confirmatory). CPU only.
Run after s1s_excess.py.
  E1. Stage sequence (Tulu SFT -> DPO -> final, OLMo SFT -> DPO -> final): mean excess and n_pos per checkpoint,
      with the final models taken from Stage 1r (s1r_excess_s1p.json); adjacent differences are described only.
  E2. Category (7) vs magnitude (9) directions per checkpoint (from s1s_excess.py output).
  E3. Null-line diagnostics per checkpoint: median slope and residual SD of the null line.
  E4. Cosine of the 16 directions between stages of the same family (same architecture), per direction.
  E5. Readout-count stability: excess recomputed leaving out each valid judgment readout in turn (if >= 4 valid).
  E6. Extrapolation check as Stage 1r X1: directions whose projection z is outside the range of the null projections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1p_rating import CONTRASTS
from s1s_excess import model_excess

FAMILIES = {
    "Tulu": ("Llama-3.1-Tulu-3-8B-SFT", "Llama-3.1-Tulu-3-8B-DPO", "Llama-3.1-Tulu-3-8B"),
    "OLMo": ("OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"),
}


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="Stage 1s s1p_rating.py outputs")
    ap.add_argument("--res_final", required=True, help="Stage 1p s1p_rating.py outputs (final models)")
    ap.add_argument("--excess", required=True, help="s1s_excess.py output")
    ap.add_argument("--excess_final", required=True, help="Stage 1r s1r_excess_s1p.json")
    ap.add_argument("--valid", required=True, help="s1s_validity.py output")
    ap.add_argument("--depth", default="0.33")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, RF = Path(a.res), Path(a.res_final)
    E, F, V = json.load(open(a.excess)), json.load(open(a.excess_final)), json.load(open(a.valid))
    out = {"note": "exploratory, not confirmatory", "E1": {}, "E2": {}, "E3": {}, "E4": {}, "E5": {}, "E6": {}}

    def mean_of(m):
        if m in E and "mean" in E[m]:
            return E[m]["mean"], E[m]["n_pos"]
        if m in F["model_mean"]:
            return F["model_mean"][m], F["model_n_pos"][m]
        return None, None

    for fam, ms in FAMILIES.items():
        seq = {m: dict(zip(("mean", "n_pos"), mean_of(m))) for m in ms}
        diffs = {}
        for x, y in zip(ms[:-1], ms[1:]):
            if seq[x]["mean"] is not None and seq[y]["mean"] is not None:
                diffs[f"{y} - {x}"] = seq[y]["mean"] - seq[x]["mean"]
        out["E1"][fam] = {"sequence": seq, "adjacent_differences": diffs}
        # E4: cosines of the directions across stages (same architecture within a family)
        D = {}
        for m in ms:
            p = (R if m in E else RF) / f"s1p_dirs_{m}_d{a.depth}.npy"
            if p.exists():
                D[m] = np.load(p)
        cs = {}
        for i, x in enumerate(ms):
            for y in ms[i + 1:]:
                if x in D and y in D and D[x].shape == D[y].shape:
                    c = cos_rows(D[x], D[y])
                    cs[f"{x} vs {y}"] = {"median": float(np.median(c)), "min": float(c.min()),
                                         "per_direction": dict(zip(CONTRASTS, map(float, c)))}
        out["E4"][fam] = cs

    for m, r in E.items():
        if "mean" not in r:
            continue
        out["E2"][m] = {"category": r["mean_category"], "magnitude": r["mean_magnitude"]}
        out["E3"][m] = {"slope_median": float(np.median(list(r["null_slope"].values()))),
                        "resid_sd_median": float(np.median(list(r["null_resid_sd"].values())))}
        vj = V[m]["valid_judgment"]
        if len(vj) >= 4:
            out["E5"][m] = {f"without_{t}": model_excess(R, m, [u for u in vj if u != t], a.depth)["mean"] for t in vj}
        j = json.load(open(R / f"s1p_{m}_d{a.depth}.json"))
        pdirs, k0 = j["grad"]["proj_dirs"], j["grad"]["proj_cols"].index("shared")
        pz, lim = [], []
        for c in CONTRASTS:
            pn = np.load(R / f"s1p_proj_null_{c}_{m}_d{a.depth}.npy")[:, k0]
            pz.append(abs((pdirs[c]["shared"] - pn.mean()) / pn.std(ddof=1)))
            lim.append(np.abs((pn - pn.mean()) / pn.std(ddof=1)).max())
        ex = np.array([r["excess"][c] for c in CONTRASTS])
        o = np.array(pz) > np.array(lim)
        out["E6"][m] = {"r_abs_projz_excess": float(np.corrcoef(pz, ex)[0, 1]), "n_outside": int(o.sum()),
                        "excess_inside": float(ex[~o].mean()) if (~o).any() else None,
                        "excess_outside": float(ex[o].mean()) if o.any() else None}

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("E1", "E2", "E3", "E5", "E6")}, indent=1))
    for fam, cs in out["E4"].items():
        for k, v in cs.items():
            print(fam, k, f"cos median {v['median']:.3f} min {v['min']:.3f}")


if __name__ == "__main__":
    main()
