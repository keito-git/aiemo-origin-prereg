"""Synthetic simulation for gate K3 (token remapping) of the P3 kill experiment.

Under lexical mediation (Sun et al.), part of the shared gradient structure is
tied to the option tokens: a mapping set m adds a component in a subspace L_m
that differs between mapping sets. lam = share of the shared structure that is
lexical (0 = none, 1 = all).

Two decision rules are compared:
  null rule     : kill if cross-mapping subspace agreement <= 95% random null
  relative rule : kill if cross-mapping agreement < ratio * within-mapping
                  agreement, computed on the same task halves.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

A_DIM, K, N_SPLITS, REPS = 30, 2, 20, 400
RATIO = 0.8


def orth(m):
    return np.linalg.qr(m)[0]


def top_k(g):
    return np.linalg.svd(g, full_matrices=False)[0][:, :K]


def agreement(u, v):
    return float(np.mean(np.linalg.svd(u.T @ v, compute_uv=False) ** 2))


def unit(x):
    return x / np.linalg.norm(x, axis=0)


def one_rep(n_disc, rho, lam, sigma, rng, q95):
    f = orth(rng.standard_normal((A_DIM, K)))
    l1 = orth(rng.standard_normal((A_DIM, K)))
    l2 = orth(rng.standard_normal((A_DIM, K)))
    spec = unit(rng.standard_normal((A_DIM, n_disc)))
    # the non-lexical shared weights are task properties, so reuse them across mappings
    shared = unit(f @ rng.standard_normal((K, n_disc)))
    g1 = (np.sqrt(rho * (1 - lam)) * shared
          + np.sqrt(rho * lam) * unit(l1 @ rng.standard_normal((K, n_disc)))
          + np.sqrt(1 - rho) * spec + np.sqrt(sigma) * unit(rng.standard_normal((A_DIM, n_disc))))
    g2 = (np.sqrt(rho * (1 - lam)) * shared
          + np.sqrt(rho * lam) * unit(l2 @ rng.standard_normal((K, n_disc)))
          + np.sqrt(1 - rho) * spec + np.sqrt(sigma) * unit(rng.standard_normal((A_DIM, n_disc))))
    cross_full = agreement(top_k(g1), top_k(g2))
    within, cross = [], []
    for _ in range(N_SPLITS):
        idx = rng.permutation(n_disc)
        a, b = idx[: n_disc // 2], idx[n_disc // 2:]
        fa = top_k(g1[:, a])
        within.append(agreement(fa, top_k(g1[:, b])))
        cross.append(agreement(fa, top_k(g2[:, b])))
    null_kill = cross_full <= q95
    rel_kill = np.mean(cross) < RATIO * np.mean(within)
    return null_kill, rel_kill, np.mean(cross) / np.mean(within)


def main():
    rng = np.random.default_rng(20261124)
    rand = [orth(rng.standard_normal((A_DIM, K))) for _ in range(2000)]
    q95 = float(np.quantile([agreement(rand[i], rand[i - 1]) for i in range(1, 2000)], 0.95))
    rows = []
    for n_disc, rho, sigma, lam in itertools.product(
            [12, 16], [0.5, 0.7], [0.25, 0.5], [0.0, 0.25, 0.5, 0.75, 1.0]):
        out = np.array([one_rep(n_disc, rho, lam, sigma, rng, q95) for _ in range(REPS)])
        rows.append({"n_disc": n_disc, "rho": rho, "sigma": sigma, "lam": lam,
                     "kill_null_rule": float(out[:, 0].mean()),
                     "kill_relative_rule": float(out[:, 1].mean()),
                     "ratio_median": float(np.median(out[:, 2]))})
        print(rows[-1], flush=True)
    Path(__file__).with_name("sim_k3_lexical_power_results.json").write_text(
        json.dumps({"q95_null": q95, "ratio_rule": RATIO, "reps": REPS, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
