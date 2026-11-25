"""Synthetic simulation for the P3 kill experiment (plan v2.1, Sec. 5).

Question: how many discovery tasks are needed so that the kill gates K1-K3
wrongly abandon a true shared k=2 functional subspace (T2-truth) at most 20%
of the time, and how often K1 wrongly passes when human VA is the true
subspace (T1-truth)?

All quantities live in the coordinates of the affective-direction span A
(dim a), because effects of affective directions depend only on P_A g.
No real data are used; parameters are grids over unknowns.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

A_DIM = 30          # number of affective directions (span A)
K = 2               # rank of the compared subspaces
N_CONF = 6          # held-out (confirmation) tasks in the kill experiment
N_SPLITS = 20       # random half-splits for K2
N_NULL = 2000       # random subspaces for the null distributions
REPS = 400


def orth(m: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(m)
    return q


def top_k(g: np.ndarray, k: int = K) -> np.ndarray:
    u, _, _ = np.linalg.svd(g, full_matrices=False)
    return u[:, :k]


def captured(u: np.ndarray, g: np.ndarray) -> float:
    """Mean fraction of squared norm of columns of g inside span(u)."""
    num = np.sum((u.T @ g) ** 2, axis=0)
    den = np.sum(g ** 2, axis=0)
    return float(np.mean(num / den))


def agreement(u: np.ndarray, v: np.ndarray) -> float:
    """Mean squared cosine of principal angles."""
    s = np.linalg.svd(u.T @ v, compute_uv=False)
    return float(np.mean(s ** 2))


def subspace_at_angle(f: np.ndarray, theta: float, rng) -> np.ndarray:
    """k-dim subspace whose principal angles to f all equal theta."""
    q = orth(np.concatenate([f, rng.standard_normal((A_DIM, K))], axis=1))
    perp = q[:, K:2 * K]
    return orth(np.cos(theta) * f + np.sin(theta) * perp)


def make_tasks(n, f, rho, sigma, rng):
    """True gradients: shared part in span(f) with fraction rho, task-specific rest.
    Returned gradients include estimation noise with variance ratio sigma."""
    shared = f @ rng.standard_normal((K, n))
    spec = rng.standard_normal((A_DIM, n))
    spec -= f @ (f.T @ spec)
    shared /= np.linalg.norm(shared, axis=0)
    spec /= np.linalg.norm(spec, axis=0)
    g = np.sqrt(rho) * shared + np.sqrt(1 - rho) * spec
    noise = rng.standard_normal((A_DIM, n))
    noise /= np.linalg.norm(noise, axis=0)
    return g, g + np.sqrt(sigma) * noise


def null_quantiles(rng):
    rand_u = [orth(rng.standard_normal((A_DIM, K))) for _ in range(N_NULL)]
    agree = [agreement(rand_u[i], rand_u[i - 1]) for i in range(1, N_NULL)]
    return rand_u, float(np.quantile(agree, 0.95))


def one_rep(n_disc, rho, sigma, theta_h, truth, rng, agree_q95):
    f_true = orth(rng.standard_normal((A_DIM, K)))
    if truth == "T2":
        h_sun = subspace_at_angle(f_true, theta_h, rng)
        h_ext = subspace_at_angle(f_true, theta_h, rng)
    else:  # T1: human VA is the true subspace (Sun version slightly rotated)
        h_ext = f_true
        h_sun = subspace_at_angle(f_true, np.deg2rad(15), rng)
    _, disc = make_tasks(n_disc, f_true, rho, sigma, rng)
    _, conf = make_tasks(N_CONF, f_true, rho, sigma, rng)
    f_hat = top_k(disc)

    # K1: F^A_2 must capture more held-out gradient than both human VA versions
    k1_kill = captured(f_hat, conf) <= max(captured(h_sun, conf), captured(h_ext, conf))

    # K2: split-half recovery must beat the random-subspace null
    cross = []
    for _ in range(N_SPLITS):
        idx = rng.permutation(n_disc)
        a, b = idx[: n_disc // 2], idx[n_disc // 2:]
        cross.append(agreement(top_k(disc[:, a]), top_k(disc[:, b])))
    k2_kill = float(np.mean(cross)) <= agree_q95

    # K3: subspace from a second token mapping (fresh estimation noise,
    # no lexical component under the hypothesis) must agree beyond the null
    disc2 = disc + np.sqrt(sigma) * rng.standard_normal(disc.shape) / np.sqrt(A_DIM)
    k3_kill = agreement(f_hat, top_k(disc2)) <= agree_q95
    return k1_kill, k2_kill, k3_kill


def main() -> None:
    rng = np.random.default_rng(20261124)
    _, agree_q95 = null_quantiles(rng)
    grid = itertools.product(
        [4, 6, 8, 10, 12, 16],          # discovery tasks
        [0.3, 0.5, 0.7],                # shared-variance fraction rho
        [0.25, 0.5, 1.0],               # estimation-noise ratio sigma
        [np.deg2rad(45), np.deg2rad(60)],  # angle human VA <-> true F (T2)
    )
    rows = []
    for n_disc, rho, sigma, theta in grid:
        res = {}
        for truth in ("T2", "T1"):
            out = np.array([one_rep(n_disc, rho, sigma, theta, truth, rng, agree_q95)
                            for _ in range(REPS)])
            res[truth] = out
        t2 = res["T2"]
        rows.append({
            "n_disc": n_disc, "rho": rho, "sigma": sigma,
            "theta_deg": round(float(np.rad2deg(theta))),
            "T2_false_kill_any": float(np.mean(t2.any(axis=1))),
            "T2_false_kill_K1": float(np.mean(t2[:, 0])),
            "T2_false_kill_K2": float(np.mean(t2[:, 1])),
            "T2_false_kill_K3": float(np.mean(t2[:, 2])),
            "T1_false_pass_K1": float(np.mean(~res["T1"][:, 0])),
        })
        print(rows[-1], flush=True)
    out = Path(__file__).with_name("sim_kill_gate_task_count_results.json")
    out.write_text(json.dumps({"agree_null_q95": agree_q95, "A_dim": A_DIM, "K": K,
                               "N_conf": N_CONF, "reps": REPS, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
