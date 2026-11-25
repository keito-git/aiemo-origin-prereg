"""P3-K step 3 (CPU): decision rules of the kill experiment, as in the prereg draft
(02_設計/04_事前登録草案_撤退実験_v2_20261124.md §4-§5, revised after external review 2) and calibrated by
../../01_シミュレーション/*.py (K1 / K1c confidence rule: sim_k1_ci_rule.py; K3: sim_k3_lexical_power.py).

Order (prereg §4):
  1. task validity  (P2 §4.2 criteria + positive control; p3k_gradients.py), in every one of --validity_maps.
     Feasibility flag: < 50% of the 28 candidates valid -> redesign (plan §5).
  2. split: discovery 16 keeping the valued (V) : non-valued (N) ratio of the valid tasks, V1 fixed in discovery, seeded;
     ALL remaining valid tasks go to confirmation (>= 6; --extra confirmation, the default). --extra drop: 22 tasks are
     drawn (ratio kept), discovery 16 / confirmation 6, the rest reported as unused. Fewer than 22 valid -> NOT_DECIDED.
  3. shared judgment gradient per task (P2 procedure, p3k_gradients.py); main gradient = mean over --main_maps
     (default: all six mappings).
  4. sigma check: per task, noise-to-signal ratio of P_A g from 100 scenario 2-splits; median over the discovery tasks
     > 0.5 -> NOT_DECIDED (add items).
  5. metric: share of |P_A g_t|^2 in the activation outlier dimensions (top 1% of mean |activation|); median over the
     discovery tasks > 0.3 -> whitened metric primary, otherwise Euclidean primary (both reported).
Kill rules (primary metric):
  K1  human VA alternative (T1). Candidates K1_HUMAN = (H_VA^Sun, H_VA^ext EmoBank, H_VA^ext NRC-VAD; prereg v2 §5).
      Sensitivity (v3.1 §8, not a gate): the same rule with H_VA^Sun-matched added (K1_SUN_MATCHED).
      H_best = the candidate with the highest mean captured fraction of the DISCOVERY gradients (chosen on discovery,
      then frozen). Per confirmation task t: Delta_t = captured_t(F^A_2) - captured_t(H_best). Kill if the one-sided
      95% upper bound mean(Delta) + t_{.95, n-1} sd(Delta) / sqrt(n) < DELTA_K = 0.05, i.e. an advantage of F^A_2 of
      DELTA_K or more is excluded. The old point rule (F^A_2 <= best of the three on confirmation) is descriptive only.
      A missing candidate -> K1 incomplete (not a registered verdict).
  K1c generic alternatives (T3-like). Candidates K1_CONTROL = (H_mag, H_ctrl), built here from the directions file:
      top-2 left singular vectors of the metric-transformed unit directions of group 'mag' (generic magnitude) and of
      group 'ctrl' (general control). Same confidence rule as K1 (best chosen on discovery, frozen). Fires ->
      KILL_T2_T3_LIKE (candidate recorded). Group 'mag' absent -> K1c incomplete (not a registered verdict).
  K2  mean over 20 random halvings of the discovery tasks of agreement(F^A_2(half a), F^A_2(half b))
      <= 95% point of the agreement of two random 2-dim subspaces (Haar in A, 2000 draws, consecutive pairs).
  K3  unused mappings 3, 4, 5, 6 (P2 used 1 and 2), shared judgment gradient from the choice readouts CAT and PAIR only
      (CTRL_CAT span removed; prereg v1 §5, §9-4). For each of the 20 discovery halvings and each ordered pair of
      distinct maps (m, m'): F from half a of map m; within = agreement with F of half b of map m; cross = agreement
      with F of half b of map m'. R = mean(cross) / mean(within) over all halvings and pairs, the relative
      mapping-invariance index (per-pair R reported). R < 0.4 -> kill; 0.4 <= R < 0.8: no kill, the main analysis uses
      the gradient averaged over all mappings. R is NOT an estimate of the lexical fraction (R is nonlinear in it).
      Unused maps missing: the available ones are used if >= 2, otherwise K3 is incomplete (INCOMPLETE).
      The all-readout R (J, JR, CAT, PAIR minus the CTRL/CTRL_CAT span) is descriptive only.
Verdict: K2 or K3 fires -> KILL_T2; else K1c -> KILL_T2_T3_LIKE; else K1 -> KILL_T2; else INCOMPLETE if K1, K1c or K3
  is incomplete; else CONTINUE. kill_rules_fired lists every rule that fired (K1 and K1c can both be recorded).
Reported, not gates:
  sensitivity  K1, K1c and K2 recomputed with choice-only (CAT/PAIR) shared gradients averaged over all mappings.
  K2b          direction-family holdout: A1 = span of groups {cat, va_sun, va_ext, emo}, A2 = span of group {app}.
               F1 = top-2 of P_A1 G_disc; captured fraction of P_A2 g_conf by orth(P_A2 F1) vs the 95% point of random
               2-dim subspaces of A2 (Haar, 2000 draws, fixed seed); and the reverse (A2 -> A1). Pass/fail per direction.
  exploratory  baselines C1-C4, F(V tasks) vs F(N tasks), label-unembedding spaces (prereg §8).
captured(U, g) = |U U^T P_A g|^2 / |P_A g|^2; agreement = mean squared cosine of the principal angles;
F^A_2 = top-2 left singular vectors of P_A G_disc. All in the residual stream of layer L (plan §1).
Whitened metric: vectors (directions, subspaces) mapped by S^{-1/2}, gradients by S^{1/2} (inner products d.g are kept),
S = activation covariance (act_stats.npz) with shrinkage.

Usage:  p3k_kill_tests.py --grad_dir <out>/<model>_L<layer> --dirs directions.npz --out verdict.json
        p3k_kill_tests.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from p3k_common import (READ, agreement, assemble_shared, captured, label_only_gradient, orth, orth_basis,
                        sigma_estimate, top_k)

TASKS = tuple([f"V{i}" for i in range(1, 17)] + [f"N{i}" for i in range(1, 13)])
BASELINES = ("C1", "C2", "C3", "C4")
ALL_MAPS = ("1", "2", "3", "4", "5", "6")     # stimuli_v1 (maps 1-4 byte-identical to stimuli_v0)
K3_MAPS = ("3", "4", "5", "6")               # mapping sets not used in P2 (P2 used 1, 2)
SEED = {"task_split": 20261221, "k2": 20261222, "k2_null": 20261223, "k3": 20261224, "k2b_null": 20261225}
N_DISC, N_CONF, K = 16, 6, 2
K1_HUMAN = ("H_VA_sun", "H_VA_ext_emobank", "H_VA_ext_nrc")   # prereg v2 §5
K1_SUN_MATCHED = K1_HUMAN + ("H_VA_sun_matched",)          # v3.1 §8: sensitivity only (external review 3)
K1_CONTROL = ("H_mag", "H_ctrl")                              # external review 2: T3-like generic alternatives
CONTROL_GROUP = {"H_mag": "mag", "H_ctrl": "ctrl"}
K2B_FAMILIES = {"A1": ("cat", "va_sun", "va_ext", "emo"), "A2": ("app",)}
DELTA_K = 0.05
N_SPLITS_K, N_NULL = 20, 2000
SIGMA_MAX, OUTLIER_FRAC, OUTLIER_SHARE_MAX = 0.5, 0.01, 0.3
R_KILL, R_CLEAN = 0.4, 0.8
FEASIBLE_FRAC = 0.5
DEFAULT_AFFECT = "emo,cat,app,va_ext,va_sun"


# ------------------------------------------------------------------ loading
def load_grad_dir(gd: Path, required: list[str], optional: list[str] = ()) -> dict:
    """Per task: entry per map, shared g per map, per-readout G per map, halves per map (same splits in every map).
    A task is kept if all `required` maps exist; `optional` maps (K3, sensitivity) are loaded where present."""
    T = {}
    for t in TASKS:
        rec = {"g": {}, "G": {}, "halves": {}, "enter": {}}
        for m in sorted(set(required) | set(optional), key=int):
            fz, fj = gd / f"shared_{t}_map{m}.npz", gd / f"shared_{t}_map{m}.json"
            if not fz.exists():
                continue
            z = np.load(fz)
            rec["g"][m], rec["G"][m], rec["halves"][m] = z["g"], z["G"], z["halves"].astype(np.float64)
            assert list(z["readouts"]) == list(READ)
            rec["enter"][m] = bool(json.load(open(fj))["enter"])
            rec.setdefault("splits", z["splits"])
            assert np.array_equal(rec["splits"], z["splits"]), "splits must be identical across maps"
        if all(m in rec["g"] for m in required):
            T[t] = rec
    base = {c: np.load(gd / f"base_{c}.npz")["g"] for c in BASELINES if (gd / f"base_{c}.npz").exists()}
    act = np.load(gd / "act_stats.npz")
    ue = dict(np.load(gd / "unembed.npz")) if (gd / "unembed.npz").exists() else {}
    return {"tasks": T, "base": base, "act": {"mean_abs": act["mean_abs"], "cov": act["cov"].astype(np.float64)},
            "unembed": ue}


def load_dirs(path: str) -> dict:
    z = np.load(path)
    return {"D": z["D"].astype(np.float64), "names": [str(x) for x in z["names"]],
            "groups": [str(x) for x in z["groups"]], "H": {k: z[k].astype(np.float64) for k in z.files
                                                            if k.startswith("H_")}}


# ------------------------------------------------------------------ steps
def split_tasks(valid: list[str], seed: int, extra: str) -> dict:
    """Discovery 16 keeping the V:N ratio of the valid tasks; V1 always in discovery.
    extra = 'confirmation' (default): discovery 16, all remaining valid tasks -> confirmation (>= 6);
    extra = 'drop': with > 22 valid tasks, 22 are drawn (ratio kept), discovery 16 / confirmation 6, rest unused;
    extra = 'discovery': confirmation 6, all extra tasks go to discovery."""
    rng = np.random.default_rng(seed)
    V = [t for t in valid if t.startswith("V")]
    N = [t for t in valid if t.startswith("N")]
    n = len(valid)
    if n < N_DISC + N_CONF:
        return {"ok": False, "n_valid": n}
    poolV = [t for t in V if t != "V1"]
    hasV1 = "V1" in V
    if extra == "confirmation":
        dV = int(round(N_DISC * len(V) / n))
        dV, dN = min(dV, len(V)), min(N_DISC - dV, len(N))
        dV = N_DISC - dN
        discV = (["V1"] if hasV1 else []) + list(rng.permutation(poolV)[: dV - hasV1])
        discN = list(rng.permutation(N)[:dN])
        disc = sorted(discV + discN)
        conf = sorted(t for t in valid if t not in disc)
        return {"ok": True, "n_valid": n, "mode": extra, "disc": disc, "conf": conf, "unused": []}
    n_tot = N_DISC + N_CONF if extra == "drop" else n
    nV = int(round(n_tot * len(V) / n))
    nV, nN = min(nV, len(V)), min(n_tot - nV, len(N))
    nV = n_tot - nN
    useV = (["V1"] if hasV1 else []) + list(rng.permutation(poolV)[: nV - hasV1])
    useN = list(rng.permutation(N)[:nN])
    cV = int(round(N_CONF * nV / n_tot))
    cV, cN = min(cV, len(useV) - ("V1" in useV)), N_CONF - cV
    confV = list(rng.permutation([t for t in useV if t != "V1"])[:cV])
    confN = list(rng.permutation(useN)[:cN])
    conf = sorted(confV + confN)
    disc = sorted(t for t in useV + useN if t not in conf)
    return {"ok": True, "n_valid": n, "mode": extra, "disc": disc, "conf": conf,
            "unused": sorted(set(valid) - set(disc) - set(conf))}


def shrink_sqrt(cov: np.ndarray, eps: float = 1e-3):
    """S^{1/2}, S^{-1/2} of the shrunk covariance S = cov + eps * tr(cov)/h * I."""
    h = len(cov)
    w, V = np.linalg.eigh(cov + eps * np.trace(cov) / h * np.eye(h))
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


class Metric:
    """Euclidean or whitened coordinates. vec(): directions and subspace bases; grad(): gradients (covectors)."""

    def __init__(self, kind: str, cov: np.ndarray | None = None):
        self.kind = kind
        if kind == "whitened":
            self.Sh, self.Sih = shrink_sqrt(cov)

    def vec(self, X):
        return X if self.kind == "euclid" else self.Sih @ X

    def grad(self, X):
        return X if self.kind == "euclid" else self.Sh @ X


def outlier_share(PAg: np.ndarray, mean_abs: np.ndarray, frac: float = OUTLIER_FRAC) -> np.ndarray:
    """Share of |P_A g|^2 (Euclidean) in the top-frac dimensions by mean |activation|. PAg: h x T."""
    n = max(1, int(np.ceil(frac * len(mean_abs))))
    O = np.argsort(-mean_abs)[:n]
    return np.sum(PAg[O] ** 2, 0) / np.sum(PAg ** 2, 0)


def k2_null_q95(a_dim: int, seed: int) -> float:
    """95% point of agreement between random k-dim subspaces in A (as the simulations: consecutive pairs of 2000)."""
    rng = np.random.default_rng(seed)
    rand = [orth(rng.standard_normal((a_dim, K))) for _ in range(N_NULL)]
    return float(np.quantile([agreement(rand[i], rand[i - 1]) for i in range(1, N_NULL)], 0.95))


def half_splits(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_SPLITS_K):
        idx = rng.permutation(n)
        out.append((idx[: n // 2], idx[n // 2:]))
    return out


def capt_each(u: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per column of g: |u^T g|^2 / |g|^2 (u orthonormal, or the A-coordinates QA^T U of an orthonormal U)."""
    return np.sum((u.T @ g) ** 2, axis=0) / np.sum(g ** 2, axis=0)


def group_basis(metric: Metric, D: np.ndarray, groups: list[str], group: str) -> np.ndarray | None:
    """H_mag / H_ctrl: top-2 left singular vectors of the metric-transformed, unit-normalized directions of a group."""
    idx = [i for i, g in enumerate(groups) if g == group]
    if not idx:
        return None
    X = metric.vec(D[idx].T)
    return top_k(X / np.linalg.norm(X, axis=0, keepdims=True), K)


def ci_rule(cands: dict, order: tuple[str, ...], F: np.ndarray, Cd: np.ndarray, Cc: np.ndarray,
            conf_names: list[str]) -> dict:
    """K1 / K1c rule (sim_k1_ci_rule.py). cands: name -> basis in A coordinates (QA^T U) or None (missing).
    Best candidate chosen on the discovery gradients and frozen; Delta_t on the confirmation tasks; kill if the
    one-sided 95% upper bound of mean(Delta) < DELTA_K. The point rule on confirmation is descriptive only."""
    present = [k for k in order if cands.get(k) is not None]
    out = {"candidates": list(order), "complete": len(present) == len(order),
           "missing": [k for k in order if cands.get(k) is None], "delta_K": DELTA_K,
           "capture_disc": {k: captured(cands[k], Cd) for k in present}}
    if not present:
        out.update(chosen=None, kill=False)
        return out
    best = max(present, key=lambda k: out["capture_disc"][k])
    d = capt_each(F, Cc) - capt_each(cands[best], Cc)
    n = len(d)
    sd = float(np.std(d, ddof=1))
    t95 = float(stats.t.ppf(0.95, n - 1))
    ub = float(d.mean() + t95 * sd / np.sqrt(n))
    cc = {k: captured(cands[k], Cc) for k in present}
    fc = captured(F, Cc)
    bc = max(cc, key=cc.get)
    out.update(chosen=best, delta_per_task=dict(zip(conf_names, d.tolist())), delta_mean=float(d.mean()),
               delta_sd=sd, n_conf=n, t95=t95, upper95=ub, kill=bool(ub < DELTA_K),
               point_rule_descriptive={"F_A_2_conf": fc, "capture_conf": cc, "best_on_conf": bc,
                                       "kill_old_rule": bool(fc <= cc[bc])})
    return out


def k3_test(c, Gk: dict) -> dict:
    """Relative mapping-invariance index R over the given maps (>= 2). Gk: map -> h x T_disc raw gradients;
    c: raw gradients -> A coordinates. R is NOT an estimate of the lexical fraction (it is nonlinear in it)."""
    maps = sorted(Gk, key=int)
    C = {m: c(Gk[m]) for m in maps}
    sp = half_splits(C[maps[0]].shape[1], SEED["k3"])
    Fa = {m: [top_k(C[m][:, i], K) for i, _ in sp] for m in maps}
    Fb = {m: [top_k(C[m][:, j], K) for _, j in sp] for m in maps}
    within = {m: np.array([agreement(a_, b_) for a_, b_ in zip(Fa[m], Fb[m])]) for m in maps}
    pairs = [(m, m2) for m in maps for m2 in maps if m != m2]
    cross = {p: np.array([agreement(a_, b_) for a_, b_ in zip(Fa[p[0]], Fb[p[1]])]) for p in pairs}
    w = float(np.mean([within[m] for m, _ in pairs]))           # mean over all halvings and ordered pairs
    x = float(np.mean([cross[p] for p in pairs]))
    R = x / w
    return {"maps": maps, "n_pairs": len(pairs), "within": w, "cross": x,
            "R_mapping_invariance": R, "R": R,
            "R_per_pair": {f"{m}->{m2}": float(np.mean(cross[(m, m2)]) / np.mean(within[m])) for m, m2 in pairs},
            "within_per_map": {m: float(np.mean(within[m])) for m in maps},
            "zone": "kill" if R < R_KILL else ("intermediate" if R < R_CLEAN else "clean"), "kill": R < R_KILL}


def haar_capture_q95(dim: int, Ct: np.ndarray, seed: int) -> float:
    """95% point of the mean captured fraction of Ct (dim x T) by Haar-random 2-dim subspaces (N_NULL draws)."""
    rng = np.random.default_rng(seed)
    return float(np.quantile([captured(orth(rng.standard_normal((dim, K))), Ct) for _ in range(N_NULL)], 0.95))


def k2b_test(metric: Metric, D: np.ndarray, groups: list[str], G_disc: np.ndarray, G_conf: np.ndarray) -> dict:
    """Secondary test K2b (reported, NOT a kill gate): direction-family holdout. F_src = top-2 of P_src G_disc;
    captured fraction of P_dst g_conf by orth(P_dst F_src) vs the Haar 95% point in dst; both directions."""
    B = {}
    for nm, gs in K2B_FAMILIES.items():
        idx = [i for i, g in enumerate(groups) if g in gs]
        B[nm] = orth_basis(metric.vec(D[idx].T)) if idx else None
    out = {"families": {k: list(v) for k, v in K2B_FAMILIES.items()},
           "dims": {k: (None if v is None else int(v.shape[1])) for k, v in B.items()}}
    if any(v is None for v in B.values()):
        out.update(computed=False, missing=[k for k, v in B.items() if v is None])
        return out
    Gd, Gc = metric.grad(G_disc), metric.grad(G_conf)
    for src, dst in (("A1", "A2"), ("A2", "A1")):
        Qs, Qt = B[src], B[dst]
        Fs = Qs @ top_k(Qs.T @ Gd, K)                       # F_src in the residual stream
        U = orth_basis(Qt.T @ Fs)                           # orth(P_dst F_src), dst coordinates
        Ct = Qt.T @ Gc                                      # P_dst g_conf, dst coordinates
        cap = captured(U, Ct)
        degenerate = Qt.shape[1] <= K                       # any 2-dim subspace captures everything
        q95 = haar_capture_q95(Qt.shape[1], Ct, SEED["k2b_null"]) if not degenerate else 1.0
        out[f"{src}_to_{dst}"] = {"captured": cap, "null_q95": q95, "rank_P_dst_F": int(U.shape[1]),
                                  "degenerate": degenerate, "pass": bool(cap > q95 and not degenerate)}
    out["computed"] = True
    return out


def analyse(metric: Metric, dirs: dict, aff: list[int], G_disc: np.ndarray, G_conf: np.ndarray, conf_names: list[str],
            Gk3: dict | None = None, Gk3_all: dict | None = None, extra: dict | None = None,
            full: bool = True) -> dict:
    """K1, K1c, K2 (and, if full, K3, K2b, exploratory) in one metric. G_*: h x T (raw gradients);
    Gk3: map -> h x T_disc choice-only gradients (primary K3), Gk3_all: same for all readouts (descriptive)."""
    D, groups = dirs["D"], dirs["groups"]
    QA = orth_basis(metric.vec(D[aff].T))                    # h x a
    a_dim = QA.shape[1]
    c = lambda G: QA.T @ metric.grad(G)                      # coordinates of P_A g in A (a x T)
    Cd, Cc = c(G_disc), c(G_conf)
    F = top_k(Cd, K)                                         # F^A_2 in A coordinates
    # candidates: |U U^T P_A g|^2 / |P_A g|^2 with U orthonormal in the (metric) residual stream -> QA^T U
    Hm = {k: orth_basis(metric.vec(v)) for k, v in dirs["H"].items() if k not in K1_CONTROL}
    Hc = {k: group_basis(metric, D, groups, g) for k, g in CONTROL_GROUP.items()}
    res = {"a_dim": a_dim, "captured_conf": {"F_A_2": captured(F, Cc), "random_expected": K / a_dim}}
    for k, U in list(Hm.items()) + [(k, U) for k, U in Hc.items() if U is not None]:
        res["captured_conf"][k] = captured(QA.T @ U, Cc)
    res["K1"] = ci_rule({k: QA.T @ Hm[k] for k in K1_HUMAN if k in Hm}, K1_HUMAN, F, Cd, Cc, conf_names)
    res["K1_sun_matched_sensitivity"] = dict(
        ci_rule({k: QA.T @ Hm[k] for k in K1_SUN_MATCHED if k in Hm}, K1_SUN_MATCHED, F, Cd, Cc, conf_names),
        gate=False)
    res["K1c"] = ci_rule({k: QA.T @ Hc[k] for k in K1_CONTROL if Hc[k] is not None}, K1_CONTROL, F, Cd, Cc,
                         conf_names)
    res["K1c"]["complete"] = Hc["H_mag"] is not None                 # 'mag' absent -> incomplete (spec)
    if Hc["H_ctrl"] is None:
        res["K1c"]["note"] = "group 'ctrl' absent: K1c computed with H_mag only"
    # K2
    sp = half_splits(Cd.shape[1], SEED["k2"])
    cross = [agreement(top_k(Cd[:, i], K), top_k(Cd[:, j], K)) for i, j in sp]
    q95 = k2_null_q95(a_dim, SEED["k2_null"])
    res["K2"] = {"agreement_mean": float(np.mean(cross)), "null_q95": q95, "kill": float(np.mean(cross)) <= q95}
    if not full:
        return res
    # K3 (relative mapping-invariance index over the unused maps)
    res["K3"] = k3_test(c, Gk3) if Gk3 and len(Gk3) >= 2 else {"kill": False, "missing": True}
    if Gk3_all and len(Gk3_all) >= 2:
        res["K3_all_readouts_descriptive"] = k3_test(c, Gk3_all)
    # K2b (secondary, not a gate)
    res["K2b"] = k2b_test(metric, D, groups, G_disc, G_conf)
    # exploratory (prereg §8): baselines and label-unembedding spaces
    extra = extra or {}
    ex = {}
    for nm, g in extra.get("baselines", {}).items():
        ex[f"captured_{nm}"] = captured(F, c(g[:, None]))
    for nm, U in extra.get("unembed", {}).items():
        B = orth_basis(metric.vec(U))
        ex[f"agreement_F_unembed_{nm}"] = agreement(QA @ F, B)
    if "disc_V" in extra and extra["disc_V"].shape[1] >= K and extra["disc_N"].shape[1] >= K:
        ex["agreement_F_V_vs_F_N"] = agreement(top_k(c(extra["disc_V"]), K), top_k(c(extra["disc_N"]), K))
    res["exploratory"] = ex
    res["F_A_2_full"] = (QA @ F).tolist() if extra.get("save_F") else None
    return res


def verdict_of(P: dict) -> tuple[str, list[str]]:
    """K2 / K3 -> KILL_T2; else K1c -> KILL_T2_T3_LIKE; else K1 -> KILL_T2; else INCOMPLETE / CONTINUE."""
    fired = [k for k in ("K1", "K1c", "K2", "K3") if P[k]["kill"]]
    if "K2" in fired or "K3" in fired:
        return "KILL_T2", fired
    if "K1c" in fired:
        return "KILL_T2_T3_LIKE", fired
    if "K1" in fired:
        return "KILL_T2", fired
    if not P["K1"]["complete"] or not P["K1c"]["complete"] or P["K3"].get("missing"):
        return "INCOMPLETE", fired
    return "CONTINUE", fired


def run(data: dict, dirs: dict, affect_groups: list[str], main_maps: list[str], validity_maps: list[str],
        extra_mode: str = "confirmation", save_F: bool = False) -> dict:
    T = data["tasks"]
    out = {"t0": time.time(), "main_maps": main_maps, "validity_maps": validity_maps, "affect_groups": affect_groups}
    # 1. validity
    valid = [t for t in TASKS if t in T and all(T[t]["enter"][m] for m in validity_maps)]
    out["valid_tasks"] = valid
    out["feasibility"] = {"frac_valid": len(valid) / len(TASKS), "redesign": len(valid) / len(TASKS) < FEASIBLE_FRAC}
    # 2. split
    sp = split_tasks(valid, SEED["task_split"], extra_mode)
    out["split"] = sp
    if not sp["ok"]:
        out["verdict"] = "NOT_DECIDED_TOO_FEW_TASKS"
        return out
    gmain = lambda t: np.mean([T[t]["g"][m] for m in main_maps], 0)
    hmain = lambda t: np.mean([T[t]["halves"][m] for m in main_maps], 0)
    G_disc = np.stack([gmain(t) for t in sp["disc"]], 1)
    G_conf = np.stack([gmain(t) for t in sp["conf"]], 1)
    aff = [i for i, g in enumerate(dirs["groups"]) if g in affect_groups]
    Daff = dirs["D"][aff]
    out["n_affect_dirs"] = len(aff)
    out["n_dirs_by_group"] = {g: dirs["groups"].count(g) for g in sorted(set(dirs["groups"]))}
    # 4. sigma (Euclidean P_A, the metric of the simulation; whitened reported)
    sig = {}
    for kind in ("euclid", "whitened"):
        M = Metric(kind, data["act"]["cov"])
        QA = orth_basis(M.vec(Daff.T))
        P = QA @ QA.T
        s_ = {}
        for t in sp["disc"] + sp["conf"]:
            s_[t] = sigma_estimate(M.grad(gmain(t)[:, None])[:, 0], np.stack([M.grad(h.T).T for h in hmain(t)]), P)
        sig[kind] = {"per_task": s_, "median_disc": float(np.median([s_[t]["sigma"] for t in sp["disc"]]))}
    out["sigma"] = sig
    out["sigma_check"] = {"median_disc": sig["euclid"]["median_disc"], "max": SIGMA_MAX,
                          "pass": sig["euclid"]["median_disc"] <= SIGMA_MAX}
    # 5. metric rule
    QA = orth_basis(Daff.T)
    sh = outlier_share(QA @ (QA.T @ G_disc), data["act"]["mean_abs"])
    primary = "whitened" if float(np.median(sh)) > OUTLIER_SHARE_MAX else "euclid"
    out["metric_rule"] = {"outlier_share_per_disc_task": dict(zip(sp["disc"], sh.tolist())),
                          "median": float(np.median(sh)), "threshold": OUTLIER_SHARE_MAX, "primary": primary}
    if not out["sigma_check"]["pass"]:
        out["verdict"] = "NOT_DECIDED_SIGMA"
        return out
    # K3 gradients (unused maps present for every discovery task). Primary: choice readouts only; descriptive: all.
    k3m = [m for m in K3_MAPS if all(m in T[t]["G"] for t in sp["disc"])]
    out["K3_maps_used"], out["K3_maps_missing"] = k3m, [m for m in K3_MAPS if m not in k3m]
    Gk3 = {m: np.stack([label_only_gradient(T[t]["G"][m]) for t in sp["disc"]], 1) for m in k3m}
    Gk3_all = {m: np.stack([T[t]["g"][m] for t in sp["disc"]], 1) for m in k3m}
    extra = {"baselines": data.get("base", {}), "save_F": save_F,
             "disc_V": np.stack([gmain(t) for t in sp["disc"] if t.startswith("V")], 1),
             "disc_N": np.stack([gmain(t) for t in sp["disc"] if t.startswith("N")], 1)}
    ue = data.get("unembed", {})
    extra["unembed"] = {k: (ue[k][1:] - ue[k][:-1]).T for k in ue if not k.endswith("_labels")}
    res = {kind: analyse(Metric(kind, data["act"]["cov"]), dirs, aff, G_disc, G_conf, sp["conf"], Gk3, Gk3_all, extra)
           for kind in ("euclid", "whitened")}
    out["results"] = res
    # sensitivity (reported, not a gate): choice-only shared gradients averaged over all mappings present
    sm = [m for m in ALL_MAPS if all(m in T[t]["G"] for t in sp["disc"] + sp["conf"])]
    gch = lambda t: np.mean([label_only_gradient(T[t]["G"][m]) for m in sm], 0)
    Sd = np.stack([gch(t) for t in sp["disc"]], 1)
    Sc = np.stack([gch(t) for t in sp["conf"]], 1)
    sens = {}
    for kind in ("euclid", "whitened"):
        r_ = analyse(Metric(kind, data["act"]["cov"]), dirs, aff, Sd, Sc, sp["conf"], full=False)
        sens[kind] = {k: r_[k] for k in ("K1", "K1c", "K2", "captured_conf")}
    out["sensitivity_choice_only_all_maps"] = {"maps": sm, "gate": False, "results": sens,
                                               "kills_primary_metric": {k: bool(sens[primary][k]["kill"])
                                                                        for k in ("K1", "K1c", "K2")}}
    P = res[primary]
    kills = {k: bool(P[k]["kill"]) for k in ("K1", "K1c", "K2", "K3")}
    out["kills_primary"] = kills
    out["kills_secondary_metric"] = {k: bool(res["whitened" if primary == "euclid" else "euclid"][k]["kill"])
                                     for k in ("K1", "K1c", "K2", "K3")}
    out["verdict"], out["kill_rules_fired"] = verdict_of(P)
    out["K1c_candidate"] = P["K1c"].get("chosen") if kills["K1c"] else None
    out["K2b_secondary"] = {d: P["K2b"].get(d, {}).get("pass") for d in ("A1_to_A2", "A2_to_A1")}
    notes = []
    if not P["K1"]["complete"]:
        notes.append(f"K1 computed without {P['K1']['missing']} (TODO): not a registered verdict")
    if not P["K1c"]["complete"]:
        notes.append("K1c: group 'mag' absent (TODO): not a registered verdict")
    if P["K3"].get("missing"):
        notes.append(f"K3 not computed (fewer than 2 unused maps present: {k3m})")
    else:
        if out["K3_maps_missing"]:
            notes.append(f"K3 computed on the available unused maps {k3m} (missing {out['K3_maps_missing']})")
        if P["K3"]["zone"] == "intermediate":
            notes.append("K3 intermediate zone: main analysis must use the gradient averaged over all mappings")
    if kills["K1"]:
        notes.append("K1: the upper bound of the advantage of F^A_2 over the human VA candidate is < DELTA_K -> T2 not"
                     " pursued; T1 route is the PI's decision (prereg §7)")
    if kills["K1c"]:
        notes.append(f"K1c: {out['K1c_candidate']} (generic alternative) is not beaten by >= DELTA_K -> T3-like")
    if kills["K1"] and kills["K1c"]:
        notes.append("K1 and K1c both fired: both recorded (kill_rules_fired)")
    out["notes"] = notes
    out["t_s"] = time.time() - out["t0"]
    return out


# ------------------------------------------------------------------ selftest (synthetic data, simulation logic)
SYN_AFFECT = ["emo", "cat", "va_ext", "va_sun", "app"]


def synth(rng, *, h=96, a_dim=30, rho=0.7, sigma=0.25, truth="T2", lam=0.0, lex_scale=1.0, n_valid=28,
          theta=np.deg2rad(45), with_sun=True, mag="random", family_shared=False, outlier=False,
          n_scen=30) -> tuple[dict, dict]:
    """Synthetic grad_dir + directions in the formats of p3k_gradients.py / p3k_directions.py.
    True gradients follow sim_kill_gate_task_count.make_tasks (in A), embedded in R^h; per-scenario readout gradients
    are generated so that the P2 shared-gradient construction and the split-half sigma estimate are exercised.
    lam: share of a map-specific lexical component (sim_k3_lexical_power) carried by the label readouts (CAT, PAIR) of
    the unused maps 3-6. A = span of 30 directions in groups emo/cat/va_ext/va_sun/app (6 each).
    mag: 'random' (3 'mag' directions random in A), 'shared' (they span the true shared subspace f), None (absent).
    3 'ctrl' directions are random in R^h. family_shared: two A1 and two A2 (app) directions lie near f (K2b)."""
    Daff = rng.standard_normal((a_dim, h))
    if outlier:                                               # two activation-outlier dims inside A
        Daff[:2] = 0.0
        Daff[0, 0] = Daff[1, 1] = 1.0
    Daff /= np.linalg.norm(Daff, axis=1, keepdims=True)
    QA = orth_basis(Daff.T)
    f = orth(rng.standard_normal((a_dim, K)))
    unit_ = lambda x: x / np.linalg.norm(x)
    groups = [SYN_AFFECT[i * len(SYN_AFFECT) // a_dim] for i in range(a_dim)]
    if family_shared:                                         # rows stay inside span(A): A itself is unchanged
        i1 = [i for i, g in enumerate(groups) if g != "app"][:2]
        i2 = [i for i, g in enumerate(groups) if g == "app"][:2]
        for r, col in zip(i1 + i2, (0, 1, 0, 1)):
            Daff[r] = unit_(QA @ (f[:, col] + 0.1 * rng.standard_normal(a_dim)))

    def at_angle(fm, th):
        q = orth(np.concatenate([fm, rng.standard_normal((a_dim, K))], 1))
        return orth(np.cos(th) * fm + np.sin(th) * q[:, K:2 * K])

    if truth == "T1":
        hum = {"H_VA_ext_emobank": f, "H_VA_sun": at_angle(f, np.deg2rad(15)),
               "H_VA_ext_nrc": at_angle(f, np.deg2rad(30))}
    else:
        hum = {k: at_angle(f, theta) for k in K1_HUMAN if k != "H_VA_sun_matched"}
    st = rng.bit_generator.state                              # v3.1 candidate drawn off the main stream
    hum["H_VA_sun_matched"] = at_angle(f, np.deg2rad(20) if truth == "T1" else theta)
    rng.bit_generator.state = st
    lexsp = {m: orth(rng.standard_normal((a_dim, K))) for m in K3_MAPS}
    perp = np.linalg.qr(np.concatenate([QA, rng.standard_normal((h, 8))], 1))[0][:, a_dim:a_dim + 8]
    ctrl_dirs = perp[:, :2].T                                 # control-readout gradients live outside A
    tasks, n_rd = {}, {"J": 2, "JR": 2, "CAT": 8, "PAIR": 12, "CTRL": 2, "CTRL_CAT": 8}
    tau = np.sqrt(120 * sigma / a_dim) if sigma > 0 else 0.0  # per scenario & readout noise sd per A-coordinate
    for ti, t in enumerate(TASKS):
        shared = unit_(f @ rng.standard_normal(K))
        spec = rng.standard_normal(a_dim)
        spec = unit_(spec - f @ (f.T @ spec))
        g_true = np.sqrt(rho) * shared + np.sqrt(1 - rho) * spec if rho > 0 else spec
        lex = {m: unit_(lexsp[m] @ rng.standard_normal(K)) for m in K3_MAPS}
        c_true = [rng.standard_normal(2) @ ctrl_dirs for _ in range(2)]
        dig_noise = {r: rng.standard_normal((n_scen, a_dim)) * tau for r in ("J", "JR", "CTRL")}
        rec = {"g": {}, "G": {}, "halves": {}, "enter": {}}
        for m in ALL_MAPS:
            S, N = {}, {}
            for r in READ:
                if r in ("CTRL", "CTRL_CAT"):
                    base = c_true[0 if r == "CTRL" else 1]
                    noise = dig_noise["CTRL"] if r == "CTRL" else rng.standard_normal((n_scen, a_dim)) * tau
                    per = base[None] + noise @ QA.T
                else:
                    sgn = -1.0 if r == "JR" else 1.0
                    gt = g_true.copy()
                    if r in ("CAT", "PAIR") and m in lex and lam > 0:
                        gt = np.sqrt(1 - lam) * g_true + np.sqrt(lam) * lex_scale * lex[m]
                    noise = dig_noise[r] if r in dig_noise else rng.standard_normal((n_scen, a_dim)) * tau
                    bias = 0.3 * rng.standard_normal(2) @ ctrl_dirs + 0.2 * perp[:, 2 + ti % 6]
                    gv = QA @ gt
                    if outlier:                       # most of the gradient mass in the outlier dims
                        gv = gv + 2.0 * np.eye(h)[0] * (1 + ti % 3) - 1.5 * np.eye(h)[1]
                    per = sgn * gv[None] + noise @ QA.T + bias[None]
                N[r] = np.full(n_scen, float(n_rd[r]))
                S[r] = per * N[r][:, None]
            sh = assemble_shared(S, N, 100, 1000 + ti)
            rec["g"][m], rec["G"][m], rec["halves"][m] = sh["g"], sh["G"], sh["halves"].astype(np.float64)
            rec["enter"][m] = ti < n_valid
        rec["sigma_true"] = sigma
        tasks[t] = rec
    mean_abs = np.ones(h)
    if outlier:
        mean_abs[:2] = 50.0
    data = {"tasks": tasks, "base": {c: rng.standard_normal((h,)) for c in BASELINES},
            "act": {"mean_abs": mean_abs, "cov": np.eye(h)}, "unembed": {}}
    H = {k: QA @ v for k, v in hum.items() if with_sun or k != "H_VA_sun"}
    rows, extra_groups = [], []
    if mag is not None:
        mix = np.array([[1.0, 0.0, 0.7071], [0.0, 1.0, 0.7071]])          # well-conditioned spanning set of f
        cm = (f @ mix + 0.05 * rng.standard_normal((a_dim, 3)) if mag == "shared"
              else rng.standard_normal((a_dim, 3)))
        rows += [unit_(QA @ cm[:, j]) for j in range(3)]
        extra_groups += ["mag"] * 3
    rows += [unit_(rng.standard_normal(h)) for _ in range(3)]
    extra_groups += ["ctrl"] * 3
    D = np.concatenate([Daff, np.stack(rows)])
    dirs = {"D": D, "names": [f"d{i}" for i in range(len(D))], "groups": groups + extra_groups, "H": H}
    return data, dirs


def selftest(reps: int) -> int:
    rng = np.random.default_rng(20261124)
    mm = list(ALL_MAPS)
    ok = True

    def rate(n, drop_maps=(), **kw):
        outs = []
        for _ in range(n):
            d_, r_ = synth(rng, **kw)
            for t in d_["tasks"].values():
                for m in drop_maps:
                    for key in ("g", "G", "halves", "enter"):
                        t[key].pop(m)
            m_ = [m for m in mm if m not in drop_maps]
            outs.append(run(d_, r_, affect_groups=SYN_AFFECT, main_maps=m_, validity_maps=m_))
        return outs

    def report(name, outs, cond, desc):
        nonlocal ok
        v = [o["verdict"] for o in outs]
        frac = {x: v.count(x) / len(v) for x in sorted(set(v))}
        good = bool(cond(outs))
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: verdicts {frac}  ({desc})", flush=True)

    kill = lambda o: o["verdict"].startswith("KILL")
    which = lambda o, k: o.get("kills_primary", {}).get(k, False)
    prim = lambda o: o["results"][o["metric_rule"]["primary"]]
    ub = lambda o, k: round(prim(o)[k]["upper95"], 3)
    # 1. T2 true, strong structure, small sigma -> mostly CONTINUE; (a) new K1 rule kill <= .05; (d) K1c kill <= .05
    o = rate(reps, rho=0.7, sigma=0.25)
    report("T2 true (rho=.7, sigma=.25)", o, lambda o: np.mean([kill(x) for x in o]) <= 0.3,
           "expect kill rate <= .3")
    report("(a) T2 true, humans at 45 deg: K1 new rule", o, lambda o: np.mean([which(x, "K1") for x in o]) <= 0.05,
           f"expect K1 kill <= .05; upper95={[ub(x, 'K1') for x in o]}; chosen="
           f"{sorted({prim(x)['K1']['chosen'] for x in o})}; old point rule kill="
           f"{np.mean([prim(x)['K1']['point_rule_descriptive']['kill_old_rule'] for x in o]):.2f}")
    report("(d) T2 true, magnitude unrelated: K1c", o, lambda o: np.mean([which(x, "K1c") for x in o]) <= 0.05,
           f"expect K1c kill <= .05; upper95={[ub(x, 'K1c') for x in o]}")
    good = all(len(x["split"]["disc"]) == 16 and len(x["split"]["conf"]) == 12 for x in o)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] confirmation mode with 28 valid: disc 16 / conf 12")
    sg = [np.median([x["sigma"]["euclid"]["per_task"][t]["sigma"] for t in x["split"]["disc"]]) for x in o]
    print(f"       sigma of the map-averaged main gradient (decision input): {np.round(sg, 3).tolist()}")
    se = [x["sensitivity_choice_only_all_maps"]["kills_primary_metric"] for x in o]
    print(f"       sensitivity (choice-only, all maps; not a gate): K1/K1c/K2 kill rates "
          f"{[round(np.mean([s[k] for s in se]), 2) for k in ('K1', 'K1c', 'K2')]}")
    # sigma estimator on single-map gradients (the synthetic sigma is defined per map)
    d_, r_ = synth(rng, rho=0.7, sigma=0.25)
    QA = orth_basis(r_["D"][[i for i, g in enumerate(r_["groups"]) if g in SYN_AFFECT]].T)
    s1 = float(np.median([sigma_estimate(d_["tasks"][t]["g"]["1"], d_["tasks"][t]["halves"]["1"], QA @ QA.T)["sigma"]
                          for t in TASKS]))
    good = 0.25 / 1.5 <= s1 <= 0.25 * 1.5
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] sigma estimator (single map): median sigma_hat {s1:.3f} (true .25)")
    # 2. (b) T1 true -> K1 kills (sim: >= .97 at n_conf = 6)
    o = rate(reps, rho=0.7, sigma=0.25, truth="T1")
    report("(b) T1 true (one human VA = F)", o, lambda o: np.mean([which(x, "K1") for x in o]) >= 0.8,
           f"expect K1 kill >= .8; chosen={[prim(x)['K1']['chosen'] for x in o]}")
    # 3. (c) shared structure inside H_mag -> KILL_T2_T3_LIKE
    o = rate(reps, rho=0.7, sigma=0.25, mag="shared")
    report("(c) shared structure in H_mag", o, lambda o: np.mean([x["verdict"] == "KILL_T2_T3_LIKE" for x in o]) >= 0.8,
           f"expect KILL_T2_T3_LIKE >= .8; candidate={sorted({str(x.get('K1c_candidate')) for x in o})}")
    # 4. no shared structure -> killed (K2 / K1)
    o = rate(reps, rho=0.0, sigma=0.25)
    report("no shared structure (rho=0)", o, lambda o: np.mean([kill(x) for x in o]) >= 0.8, "expect kill >= .8")
    # 5. (e) lexical mediation, K3 over the 4 unused maps
    o = rate(reps, rho=0.7, sigma=0.25, lam=1.0, lex_scale=3.0)
    report("(e) lexical dominant (label readouts, 4 unused maps)", o,
           lambda o: np.mean([which(x, "K3") for x in o]) >= 0.7 and all(prim(x)["K3"]["n_pairs"] == 12 for x in o),
           "expect K3 kill >= .7, 12 ordered pairs; R=" + str([round(prim(x)["K3"]["R"], 2) for x in o]))
    # lexical component only in CAT/PAIR at the size of the non-lexical part: primary (choice-only) K3 must kill,
    # the all-readout R must be diluted (larger) because J/JR are identical across mappings
    o = rate(reps, rho=0.7, sigma=0.25, lam=1.0, lex_scale=1.0)
    Rp = [prim(x)["K3"]["R"] for x in o]
    Ra = [prim(x)["K3_all_readouts_descriptive"]["R"] for x in o]
    report("lexical only in CAT/PAIR (dilution)", o,
           # dilution (4 unused maps): all-readout R above the primary R in every rep and the all-readout version
           # kills clearly less often (the old 2-map check 'median R_all >= .4' gave median .395 here)
           lambda o: (np.mean([which(x, "K3") for x in o]) >= 0.7 and len(Ra) == len(o)
                      and all(a_ > p_ for a_, p_ in zip(Ra, Rp))
                      and np.mean(np.array(Ra) < R_KILL) <= np.mean(np.array(Rp) < R_KILL) - 0.3),
           f"expect primary K3 kill >= .7 and all-readout R diluted (> primary in every rep, kill rate >= .3 lower; "
           f"all-readout kill rate {np.mean(np.array(Ra) < R_KILL):.2f}, median R_all {np.median(Ra):.2f}); "
           f"R_primary={np.round(Rp, 2).tolist()} R_all={np.round(Ra, 2).tolist()}")
    o = rate(reps, rho=0.7, sigma=0.25, lam=0.0)
    report("(e) no lexical component", o, lambda o: np.mean([which(x, "K3") for x in o]) <= 0.1,
           "expect K3 kill <= .1; R=" + str([round(prim(x)["K3"]["R"], 2) for x in o]))
    # K3 with missing unused maps
    o = rate(2, drop_maps=("5", "6"), rho=0.7, sigma=0.25)
    report("maps 5, 6 missing", o, lambda o: all(x["K3_maps_used"] == ["3", "4"] and not prim(x)["K3"].get("missing")
                                                 and prim(x)["K3"]["n_pairs"] == 2 for x in o),
           "expect K3 on maps 3, 4 (2 ordered pairs)")
    o = rate(2, drop_maps=("4", "5", "6"), rho=0.7, sigma=0.25)
    report("only map 3 unused", o, lambda o: all(x["verdict"] in ("INCOMPLETE", "KILL_T2", "KILL_T2_T3_LIKE")
                                                 and prim(x)["K3"].get("missing") for x in o),
           "expect K3 missing -> INCOMPLETE (or KILL by another rule)")
    # 6. (f) K2b runs and passes when the shared structure spans both direction families
    o = rate(reps, rho=0.7, sigma=0.25, family_shared=True)
    p12 = [prim(x)["K2b"]["A1_to_A2"]["pass"] for x in o]
    p21 = [prim(x)["K2b"]["A2_to_A1"]["pass"] for x in o]
    report("(f) K2b, shared structure spans both families", o,
           lambda o: all(prim(x)["K2b"]["computed"] for x in o) and np.mean(p12) >= 0.8 and np.mean(p21) >= 0.8,
           f"expect K2b pass >= .8 both directions; pass A1->A2 {np.mean(p12):.2f}, A2->A1 {np.mean(p21):.2f}; "
           f"capt/q95 A1->A2 {[(round(prim(x)['K2b']['A1_to_A2']['captured'], 2), round(prim(x)['K2b']['A1_to_A2']['null_q95'], 2)) for x in o[:3]]}")
    # 7. large sigma -> not decided
    o = rate(max(3, reps // 2), rho=0.7, sigma=2.0)
    report("sigma = 2", o, lambda o: all(x["verdict"] == "NOT_DECIDED_SIGMA" for x in o), "expect NOT_DECIDED_SIGMA")
    # 8. too few valid tasks
    o = rate(2, n_valid=20)
    report("20 valid tasks", o, lambda o: all(x["verdict"] == "NOT_DECIDED_TOO_FEW_TASKS" and
                                              x["feasibility"]["frac_valid"] < 1 for x in o),
           "expect NOT_DECIDED_TOO_FEW_TASKS")
    o = rate(2, n_valid=12)
    report("12 valid tasks", o, lambda o: all(x["feasibility"]["redesign"] for x in o), "expect redesign flag")
    # 9. missing candidates -> not a registered verdict
    o = rate(2, rho=0.7, sigma=0.25, with_sun=False)
    report("H_VA_sun missing", o, lambda o: all(x["verdict"] in ("INCOMPLETE", "KILL_T2", "KILL_T2_T3_LIKE") and
                                                not x["results"]["euclid"]["K1"]["complete"] for x in o),
           "expect INCOMPLETE (or KILL by another rule)")
    o = rate(2, rho=0.7, sigma=0.25, mag=None)
    report("group 'mag' missing", o, lambda o: all(x["verdict"] in ("INCOMPLETE", "KILL_T2", "KILL_T2_T3_LIKE") and
                                                   not x["results"]["euclid"]["K1c"]["complete"] for x in o),
           "expect INCOMPLETE (or KILL by another rule)")
    # 10. metric rule
    o = rate(2, rho=0.7, sigma=0.25, outlier=True)
    report("outlier dims", o, lambda o: all(x["metric_rule"]["primary"] == "whitened" for x in o),
           "expect whitened primary; median share " + str([round(x["metric_rule"]["median"], 2) for x in o]))
    o = rate(2, rho=0.7, sigma=0.25)
    report("no outlier dims", o, lambda o: all(x["metric_rule"]["primary"] == "euclid" for x in o),
           "expect euclid primary")
    # 11. task split
    s = split_tasks(list(TASKS), 1, "drop")
    good = (len(s["disc"]) == 16 and len(s["conf"]) == 6 and "V1" in s["disc"] and not set(s["disc"]) & set(s["conf"])
            and sum(t.startswith("V") for t in s["disc"] + s["conf"]) == round(22 * 16 / 28))
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] task split (drop): disc {s['disc']} conf {s['conf']}")
    good = True
    for n_ in (22, 25, 28):
        vt = [t for t in TASKS if t.startswith("V")][: round(n_ * 16 / 28)] + \
             [t for t in TASKS if t.startswith("N")][: n_ - round(n_ * 16 / 28)]
        s = split_tasks(vt, SEED["task_split"], "confirmation")
        good &= (len(s["disc"]) == 16 and len(s["conf"]) == n_ - 16 and "V1" in s["disc"] and not s["unused"]
                 and not set(s["disc"]) & set(s["conf"]) and set(s["disc"]) | set(s["conf"]) == set(vt)
                 and sum(t.startswith("V") for t in s["disc"]) == round(16 * sum(t.startswith("V") for t in vt) / n_))
    ok &= good
    s = split_tasks(list(TASKS), SEED["task_split"], "confirmation")
    print(f"[{'PASS' if good else 'FAIL'}] task split (confirmation, n = 22/25/28): 28 -> disc {s['disc']} "
          f"conf {s['conf']}")
    print("SELFTEST_OK" if ok else "SELFTEST_FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad_dir")
    ap.add_argument("--dirs")
    ap.add_argument("--out")
    ap.add_argument("--affect_groups", default=DEFAULT_AFFECT)
    ap.add_argument("--main_maps", default=",".join(ALL_MAPS), help="maps averaged for K1/K1c/K2")
    ap.add_argument("--validity_maps", default=",".join(ALL_MAPS), help="a task is valid if it enters in every one")
    ap.add_argument("--extra", choices=("confirmation", "drop", "discovery"), default="confirmation")
    ap.add_argument("--save_F", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.reps))
    req = sorted(set(a.main_maps.split(",")) | set(a.validity_maps.split(",")), key=int)
    data = load_grad_dir(Path(a.grad_dir), req, list(ALL_MAPS))
    res = run(data, load_dirs(a.dirs), a.affect_groups.split(","), a.main_maps.split(","), a.validity_maps.split(","),
              a.extra, a.save_F)
    res["inputs"] = {"grad_dir": a.grad_dir, "dirs": a.dirs}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res.get(k) for k in ("verdict", "kill_rules_fired", "K1c_candidate", "split", "sigma_check",
                                              "metric_rule", "kills_primary", "K2b_secondary", "notes")},
                     default=float)[:4000])


if __name__ == "__main__":
    main()
