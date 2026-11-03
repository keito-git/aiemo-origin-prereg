"""Stage 1o power of I1 (sum over models of the min z over the six judgment readouts), I1 without L, and I4 (CAT_S2 +
PAIR only), by simulation (2026-11-03, after internal review R3 of design 11). The correlation between the
sign-aligned readouts under random directions is taken from the pilot-3 nulls (03_解析/stage1o_gradpos, mean over the
three models); every judgment readout is shifted by the same true z (1, 1.5, 2, 3); alpha = .05 one-sided.
Usage: python3 code/s1o_power_sim.py 03_解析/stage1o_gradpos"""
import glob
import sys

import numpy as np

R = ["S", "HARSH", "L", "U_bright", "U_fam", "CAT_S", "CAT_U", "CAT_R", "CAT_S2", "PAIR"]
J = ["S", "HARSH", "L", "CAT_S", "CAT_S2", "PAIR"]
SGN = np.array([1, 1, -1, 1, 1, 1.0])
idx = [R.index(t) for t in J]
C = np.mean([np.corrcoef((np.load(f)[:, idx] * SGN).T)
             for f in sorted(glob.glob(f"{sys.argv[1]}/s1o_gradpos_null_*.npy"))], 0)
Lc = np.linalg.cholesky(C + 1e-6 * np.eye(6))
rng = np.random.default_rng(1)
stat = lambda z, cols: z[..., cols].min(-1).sum(-1)
for name, cols in (("I1", list(range(6))), ("I1_noL", [0, 1, 3, 4, 5]), ("I4", [4, 5])):
    for M in (3, 4, 6):
        crit = np.quantile(stat(rng.standard_normal((20000, M, 6)) @ Lc.T, cols), .95)
        pw = [float((stat(rng.standard_normal((20000, M, 6)) @ Lc.T + d, cols) > crit).mean())
              for d in (1, 1.5, 2, 3)]
        print(name, "M", M, "power at z = 1, 1.5, 2, 3:", np.round(pw, 2).tolist())
