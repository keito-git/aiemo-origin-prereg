"""Synthetic simulation for the revised K1 of the P3 kill experiment (external review 2, item 1).

Old K1: kill if capture(F^A_2) <= max over 3 human candidates of capture, all on the confirmation tasks
        (point estimate; best-of-three chosen on the confirmation tasks).
New K1: the human candidate is chosen on the DISCOVERY tasks and frozen. Per confirmation task t,
        Delta_t = capture_t(F) - capture_t(H_best). Kill if the one-sided 95% upper bound of mean(Delta)
        (t distribution, df = n_conf - 1) is below delta_K, i.e. a substantive advantage of F is excluded.

Truths simulated (A coordinates, dim 30, k = 2):
  T2      human candidates at angle theta (45 / 60 deg) to the true shared subspace
  T2weak  human candidates at 25 deg (F is better, but only a little)
  T1      one human candidate equals the true subspace, the others at 15 / 30 deg
Reported: kill rate of the old and the new rule; overall rate with K2 (unchanged).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from sim_kill_gate_task_count import (A_DIM, K, N_SPLITS, agreement, make_tasks, null_quantiles, orth,
                                      subspace_at_angle, top_k)

REPS = 400


def capt_each(u, g):
    return np.sum((u.T @ g) ** 2, axis=0) / np.sum(g ** 2, axis=0)


def one_rep(n_disc, n_conf, rho, sigma, truth, deltas, rng, q95):
    f = orth(rng.standard_normal((A_DIM, K)))
    if truth == "T1":
        hs = [f, subspace_at_angle(f, np.deg2rad(15), rng), subspace_at_angle(f, np.deg2rad(30), rng)]
    else:
        th = {"T2_45": 45, "T2_60": 60, "T2weak": 25}[truth]
        hs = [subspace_at_angle(f, np.deg2rad(th), rng) for _ in range(3)]
    _, disc = make_tasks(n_disc, f, rho, sigma, rng)
    _, conf = make_tasks(n_conf, f, rho, sigma, rng)
    fh = top_k(disc)
    old = capt_each(fh, conf).mean() <= max(capt_each(h, conf).mean() for h in hs)
    hb = hs[int(np.argmax([capt_each(h, disc).mean() for h in hs]))]     # chosen on discovery, frozen
    d = capt_each(fh, conf) - capt_each(hb, conf)
    ub = d.mean() + stats.t.ppf(0.95, n_conf - 1) * d.std(ddof=1) / np.sqrt(n_conf)
    new = {dk: bool(ub < dk) for dk in deltas}
    cross = []
    for _ in range(N_SPLITS):
        idx = rng.permutation(n_disc)
        cross.append(agreement(top_k(disc[:, idx[: n_disc // 2]]), top_k(disc[:, idx[n_disc // 2:]])))
    k2 = float(np.mean(cross)) <= q95
    return old, new, k2, float(d.mean())


def main():
    rng = np.random.default_rng(20261125)
    _, q95 = null_quantiles(rng)
    deltas = (0.0, 0.02, 0.05, 0.10)
    rows = []
    for n_conf, rho, sigma, truth in itertools.product([6, 10], [0.5, 0.7], [0.25, 0.5],
                                                       ["T2_45", "T2_60", "T2weak", "T1"]):
        out = [one_rep(16, n_conf, rho, sigma, truth, deltas, rng, q95) for _ in range(REPS)]
        row = {"n_conf": n_conf, "rho": rho, "sigma": sigma, "truth": truth,
               "old_K1_kill": float(np.mean([o[0] for o in out])),
               "K2_kill": float(np.mean([o[2] for o in out])),
               "mean_delta": float(np.mean([o[3] for o in out]))}
        for dk in deltas:
            row[f"new_K1_kill_d{dk}"] = float(np.mean([o[1][dk] for o in out]))
            row[f"any_new_d{dk}"] = float(np.mean([o[1][dk] or o[2] for o in out]))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    Path(__file__).with_name("sim_k1_ci_rule_results.json").write_text(
        json.dumps({"reps": REPS, "n_disc": 16, "deltas": deltas, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
