"""Stage 1l sensitivity (not registered): L1 with the paired-null draws whose cosine missed the target removed.
Reads the per-draw cosines from diag/s1l_pair_cos_diag.py (row i = row i of s1l_null_pair_neg/pos).
A draw index is kept only if its cosine error is <= tol in every gated model, so the summed null stays index-paired.
Run inside the copied results folder; pass the diagnostic folder with --diag."""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True)
ap.add_argument("--tol", type=float, default=0.01)
a = ap.parse_args()
D = Path(a.diag)
G = json.load(open("s1l_pooled_tests.json"))["gate"]
eff = lambda m, n: json.load(open(f"s1l_{m}_anger_d0.33.json"))["readout"]["R9k"]["effects"][n]
nul = lambda m, f: np.load(f"s1l_null_{f}_{m}_anger_d0.33.npy")
pu = lambda z, x: (np.sum(z >= x) + 1) / (len(z) + 1)

keep = None
miss = {}
for m in G:
    rep = json.load(open(D / f"s1l_pair_cos_{m}_anger_d0.33.json"))
    err = np.load(D / f"s1l_pair_cos_{m}_anger_d0.33.npy") - rep["cos_target"]
    ok = np.abs(err) <= a.tol
    miss[m] = int((~ok).sum())
    keep = ok if keep is None else keep & ok


def stat(terms, idx):
    x = sum(s * (eff(m, n)["dslope_S"] - eff(m, n)["dslope_C"]) if c == "J" else s * eff(m, n)[c]
            for m in G for s, c, n, f in terms)
    z = sum(s * ((nul(m, f)[:, 2] - nul(m, f)[:, 3]) if c == "J" else nul(m, f)[:, 0])[idx]
            for m in G for s, c, n, f in terms)
    return {"obs": float(x), "null_mean": float(z.mean()), "null_sd": float(z.std(ddof=1)), "p_upper": float(pu(z, x)),
            "n": int(len(z))}


L1 = [(1, "J", "pc1_neg", "pair_neg"), (-1, "J", "pc1_pos", "pair_pos")]
L1e = [(1, "dE_S", "pc1_neg", "pair_neg"), (-1, "dE_S", "pc1_pos", "pair_pos")]
allidx = np.ones_like(keep)
out = {"tol": a.tol, "misses_per_model": miss, "n_kept": int(keep.sum()),
       "L1_all": stat(L1, allidx), "L1_kept": stat(L1, keep),
       "L1e_all": stat(L1e, allidx), "L1e_kept": stat(L1e, keep)}
Path("explore").mkdir(exist_ok=True)
json.dump(out, open("explore/s1l_pair_sensitivity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
