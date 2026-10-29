"""Stage 1l exploratory (not registered) robustness: per-model p and leave-one-model-out for L1 and L2,
and the S / C split of J for each direction. Run inside the copied results folder; reads s1l_pooled_tests.json."""
import json
import numpy as np

COLS = ("dE_S", "dE_C", "dslope_S", "dslope_C", "dE_S_sev_minus_mild", "dslope_S_sev_minus_mild")
NULL_FILE = {"pc1_neg": "pair_neg", "pc1_pos": "pair_pos", "int_neg": "int_neg", "int_pos": "int_pos",
             "val_pc1": "val_pc1"}
o = json.load(open("s1l_pooled_tests.json"))
G, GN = o["gate"], o["gate_int_neg"]
eff = lambda m, n: json.load(open(f"s1l_{m}_anger_d0.33.json"))["readout"]["R9k"]["effects"][n]
nul = lambda m, n: np.load(f"s1l_null_{NULL_FILE[n]}_{m}_anger_d0.33.npy")
pu = lambda z, x: (np.sum(z >= x) + 1) / (len(z) + 1)


def J(m, n):
    e, z = eff(m, n), nul(m, n)
    return e["dslope_S"] - e["dslope_C"], z[:, 2] - z[:, 3]


def L1(m):
    a, b = J(m, "pc1_neg"), J(m, "pc1_pos")
    return a[0] - b[0], a[1] - b[1]   # paired null: draw i with draw i


def L2(m):
    return J(m, "int_neg")


TESTS = {"L1": (L1, G), "L2": (L2, GN)}
out = {"per_model": {}, "loo": {}, "S_C_split": {}}
for f, (fn, ms) in TESTS.items():
    for m in ms:
        x, z = fn(m)
        out["per_model"].setdefault(m, {})[f] = {"obs": float(x), "p_upper": float(pu(z, x))}
    for drop in ms:
        rest = [m for m in ms if m != drop]
        if len(rest) < 2:
            continue
        n = min(len(fn(m)[1]) for m in rest)
        x = sum(fn(m)[0] for m in rest)
        z = sum(fn(m)[1][:n] for m in rest)
        out["loo"].setdefault(drop, {})[f] = {"obs": float(x), "p_upper": float(pu(z, x))}
# which question moves: summed dslope_S and dslope_C per direction over the gated models
for n in NULL_FILE:
    out["S_C_split"][n] = {c: float(sum(eff(m, n)[c] for m in G)) for c in ("dslope_S", "dslope_C", "dE_S", "dE_C")}
json.dump(out, open("explore/s1l_explore.json", "w"), indent=1)
print(json.dumps(out, indent=1))
