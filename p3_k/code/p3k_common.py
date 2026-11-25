"""P3-K shared logic (CPU only; no torch import). Used by p3k_gradients.py, p3k_directions.py and p3k_kill_tests.py.

Contents
  * readout definitions and score weights (copied from P2 s1o_rating.py: digit = OLS slope of log p over k = 1..9;
    ranked options = slope over the semantic rank, all rotations; pair = log-odds of the heavier option, both orders)
  * no-steering validity criteria (copied from P2 s1o_pooled_tests.validity, readout names mapped
    S -> J, L -> JR, HARSH -> (none), CAT_S -> CAT, PAIR -> PAIR, U_bright -> CTRL, CAT_U -> CTRL_CAT)
  * task entry rule (P2 model entry rule applied per task)
  * shared judgment gradient (P2 body.tex Eq. 3-5: remove the span of the control gradients, sign-align the reverse-keyed
    readout, average without normalization)
  * scenario 2-splits, sigma estimate, subspace utilities (identical to ../../01_シミュレーション/*.py),
    variance-matched random directions (copied from P2 s1d_word_specificity.fast_var_matched)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

# ------------------------------------------------------------------ readouts
JUDG = ("J", "JR", "CAT", "PAIR")           # judgment readouts (P2: S, L, CAT_S, PAIR; HARSH has no P3 counterpart)
CTRL = ("CTRL", "CTRL_CAT")                 # control readouts (P2: U_bright, CAT_U)
READ = JUDG + CTRL
DIGIT_R = ("J", "JR", "CTRL")               # map-independent (answer = digit)
LABEL_R = ("CAT", "PAIR", "CTRL_CAT")       # map-dependent (answer = option label)
DIG, CHO = ("J", "JR"), ("CAT", "PAIR")     # entry rule: >= 1 rating and >= 1 choice readout
SIGN = {"J": 1.0, "JR": -1.0, "CAT": 1.0, "PAIR": 1.0}   # JR asks for leniency-type (reverse-keyed) judgments
PAIR_ROTS = ("12a", "12b", "23a", "23b", "34a", "34b")
CAT_ROTS = ("0", "1", "2", "3")
SEVS = ("weak", "strong")                   # canonical order: weak block first (P2 summ(): first half mild)
K9 = np.arange(1, 10)
W9 = (K9 - K9.mean()) / ((K9 - K9.mean()) @ (K9 - K9.mean()))
MAPS = ("1", "2", "3", "4", "5", "6")      # stimuli_v1 (maps 5, 6 added 2026-11-25; 1-4 identical to v0)
K3_MAPS = ("3", "4", "5", "6")              # mapping sets not used in P2 (P2 used 1, 2)


def scen_num(scen: str) -> int:
    """'V1_07' -> 7 (P2 used int(scen[1:]) on 'L07')."""
    return int(scen.split("_")[-1])


def rot_order(readout: str) -> tuple[str, ...]:
    if readout in ("CAT", "CTRL_CAT"):
        return CAT_ROTS
    if readout == "PAIR":
        return PAIR_ROTS
    return ("",)


def canonical_items(rows: list[dict], readout: str) -> list[dict]:
    """Items of one (task, readout) ordered rotation x severity (weak, strong) x scenario; asserts a full design."""
    scens = sorted({x["scen"] for x in rows})
    key = {(x["rot"], x["sev"], x["scen"]): x for x in rows}
    out = []
    for r in rot_order(readout):
        for s in SEVS:
            for sc in scens:
                assert (r, s, sc) in key, (readout, r, s, sc)
                out.append(key[(r, s, sc)])
    assert len(out) == len(rows), (readout, len(out), len(rows))
    return out


def score_weights(items: list[dict], readout: str) -> tuple[np.ndarray, list[list[str]]]:
    """Per-item weights over the answer choices (score = w . log p, p renormalized over the choices) and the
    choice strings in position order. Exactly the P2 s1o_rating.py weights."""
    if readout in DIGIT_R:
        return np.tile(W9, (len(items), 1)), [[str(k) for k in K9] for _ in items]
    labs = [x["labels"].split("|") for x in items]
    rk = np.array([[int(c) for c in x["ranks"]] for x in items], dtype=float)
    if readout in ("CAT", "CTRL_CAT"):
        assert rk.shape[1] == 4
        return (rk - 2.5) / 5.0, labs                       # OLS slope over rank 1..4
    if readout == "PAIR":
        assert rk.shape[1] == 2
        return np.where(rk == rk.max(1, keepdims=True), 1.0, -1.0), labs   # log-odds of the heavier option
    raise ValueError(readout)


def baseline_weights(items: list[dict]) -> tuple[np.ndarray, list[list[str]]]:
    """C1-C4 (exploratory): C1/C2 digit rating, C3/C4 pair with ranks ('21' = correct option first)."""
    if items[0]["readout"] == "J":
        return np.tile(W9, (len(items), 1)), [[str(k) for k in K9] for _ in items]
    rk = np.array([[int(c) for c in x["ranks"]] for x in items], dtype=float)
    return np.where(rk == rk.max(1, keepdims=True), 1.0, -1.0), [x["labels"].split("|") for x in items]


# ------------------------------------------------------------------ validity (P2 s1o_pooled_tests.py)
def summ(x, xS) -> dict:
    """x, xS: scores per (severity, scenario), weak block then strong block (P2: mild then severe)."""
    x, xS = np.asarray(x, float), np.asarray(xS, float)
    n = len(x) // 2
    d = x[n:] - x[:n]
    return {"severity_effect": float(d.mean()), "frac_pos": float((d > 0).mean()), "frac_neg": float((d < 0).mean()),
            "rho_mild_severe": float(spearmanr(x[:n], x[n:])[0]), "rho_with_S": float(spearmanr(x, xS)[0])}


def rot_cons(E) -> float:
    E = np.asarray(E, float)
    return float(np.mean([spearmanr(E[i], E[j])[0] for i, j in combinations(range(len(E)), 2)]))


def base_scores(readout: str, logp: np.ndarray, W: np.ndarray, n_unit: int) -> dict:
    """Baseline (no steering) scores per (severity, scenario) unit, as in s1o_rating.py:
    digit -> E[digit]; ranked options -> expected rank, mean over rotations (and by rotation);
    pair -> log-odds of the heavier option, mean over the 6 order x pair rotations (and by rotation)."""
    if readout in DIGIT_R:
        return {"score": (np.exp(logp) @ K9).tolist()}
    if readout in ("CAT", "CTRL_CAT"):
        rk = W * 5.0 + 2.5
        E = (np.exp(logp) * rk).sum(1).reshape(4, n_unit)
        return {"score": E.mean(0).tolist(), "by_rot": E.tolist()}
    lo = (logp * W).sum(1).reshape(6, n_unit)
    return {"score": lo.mean(0).tolist(), "by_rot": lo.tolist()}


def validity_readout(readout: str, sc: dict, ref_J: list[float]) -> tuple[dict, bool]:
    """P2 criteria (s1o_pooled_tests.validity), with the P3 reference rating = J (P2: S).
    J: effect > 1, frac_pos >= .8 | JR: effect < -.5, frac_neg >= .8, rho(., J) < -.5
    CAT: effect > 1, rotation consistency > .8, rho(., J) > .7 | PAIR: effect > 1, frac_pos >= .8, rho(., J) > .7
    CTRL: |effect| < 1.2, rho(weak, strong) > .5, |rho(., J)| < .3
    CTRL_CAT: |effect| < .5, rotation consistency > .5, |rho(., J)| < .3"""
    s = summ(sc["score"], ref_J)
    if "by_rot" in sc and readout in ("CAT", "CTRL_CAT"):
        s["rot_consistency"] = rot_cons(sc["by_rot"])
    if readout == "PAIR":
        P = np.asarray(sc["by_rot"])
        s["order_consistency"] = float(np.mean([spearmanr(P[2 * k], P[2 * k + 1])[0] for k in range(3)]))
    v = {"J": lambda: s["severity_effect"] > 1 and s["frac_pos"] >= .8,
         "JR": lambda: s["severity_effect"] < -.5 and s["frac_neg"] >= .8 and s["rho_with_S"] < -.5,
         "CAT": lambda: s["severity_effect"] > 1 and s["rot_consistency"] > .8 and s["rho_with_S"] > .7,
         "PAIR": lambda: s["severity_effect"] > 1 and s["frac_pos"] >= .8 and s["rho_with_S"] > .7,
         "CTRL": lambda: abs(s["severity_effect"]) < 1.2 and s["rho_mild_severe"] > .5 and abs(s["rho_with_S"]) < .3,
         "CTRL_CAT": lambda: (abs(s["severity_effect"]) < .5 and s["rot_consistency"] > .5
                              and abs(s["rho_with_S"]) < .3)}[readout]()
    return s, bool(v)


def task_entry(valid: dict[str, bool]) -> bool:
    """P2 model entry rule applied to one task: >= 3 valid judgment readouts with >= 1 rating (J, JR) and >= 1 choice
    (CAT, PAIR), and >= 1 valid control readout. valid[t] must already include the gradient positive control."""
    vj = [t for t in JUDG if valid.get(t, False)]
    return (len(vj) >= 3 and any(t in DIG for t in vj) and any(t in CHO for t in vj)
            and any(valid.get(t, False) for t in CTRL))


# ------------------------------------------------------------------ shared judgment gradient (P2 Eq. 3-5)
def shared_gradient(G: dict[str, np.ndarray]) -> np.ndarray:
    """G[t] = mean gradient of readout t (all judgment and control readouts). Residualize each judgment gradient on the
    span of the control gradients, sign-align, average without normalization (P2 s1o_rating.py 'shared')."""
    Q = np.linalg.qr(np.stack([G[t] for t in CTRL]).T)[0]
    r = [SIGN[t] * (G[t] - Q @ (Q.T @ G[t])) for t in JUDG]
    return np.mean(r, 0)


def readout_means(S: dict[str, np.ndarray], N: dict[str, np.ndarray], sel: np.ndarray | None = None) -> dict:
    """S[t]: per-scenario sums of item gradients (n_scen x h); N[t]: item counts per scenario.
    Returns the mean gradient per readout over the selected scenarios (all if sel is None)."""
    out = {}
    for t in S:
        m = np.ones(len(N[t]), bool) if sel is None else sel
        out[t] = S[t][m].sum(0) / N[t][m].sum()
    return out


def scenario_splits(n_scen: int, n_splits: int, seed: int) -> np.ndarray:
    """n_splits random 2-splits of the scenarios (boolean masks for half a; half b is the complement).
    Both items of a pair and all rotations of a scenario stay together."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_splits, n_scen), bool)
    for i in range(n_splits):
        out[i, rng.permutation(n_scen)[: n_scen // 2]] = True
    return out


def split_halves(S, N, splits: np.ndarray) -> np.ndarray:
    """Shared gradient of each half for every split: array (n_splits, 2, h)."""
    out = []
    for m in splits:
        out.append(np.stack([shared_gradient(readout_means(S, N, m)), shared_gradient(readout_means(S, N, ~m))]))
    return np.stack(out)


def assemble_shared(S: dict, N: dict, n_splits: int, seed: int) -> dict:
    """Everything p3k_kill_tests.py needs from one (task, map): readout means, shared gradient, split halves."""
    G = readout_means(S, N)
    splits = scenario_splits(len(next(iter(N.values()))), n_splits, seed)
    return {"g": shared_gradient(G), "G": np.stack([G[t] for t in READ]), "readouts": np.array(READ),
            "halves": split_halves(S, N, splits).astype(np.float32), "splits": splits}


def label_only_gradient(G: np.ndarray) -> np.ndarray:
    """Primary K3 gradient (prereg v1 §5, §9-4): shared judgment gradient from the choice readouts (CAT, PAIR) only,
    residualized on the span of the CTRL_CAT gradient (the choice-format control), sign-aligned, averaged without
    normalization. G rows in READ order. The digit readouts (J, JR, CTRL) answer with the same digit tokens in every
    mapping, so they are left out (their lexical route cannot be tested by remapping)."""
    Gd = dict(zip(READ, G))
    Q = np.linalg.qr(Gd["CTRL_CAT"][:, None])[0]
    return np.mean([SIGN[t] * (Gd[t] - Q @ (Q.T @ Gd[t])) for t in ("CAT", "PAIR")], 0)


def sigma_estimate(g_full: np.ndarray, halves: np.ndarray, P: np.ndarray | None = None) -> dict:
    """Noise-to-signal ratio sigma of one task's gradient estimate (the sigma of the simulation:
    g_hat = g + sqrt(sigma) * noise with |noise| = |g|).
    With equal halves, E|g_a - g_b|^2 = 4 v, v = noise variance of the full estimate; signal = |g_hat|^2 - v.
    P: optional projector (P_A, possibly in the whitened metric) applied first."""
    f = (lambda x: x) if P is None else (lambda x: x @ P.T)
    d = f(halves[:, 0] - halves[:, 1])
    v = float(np.mean(np.sum(d ** 2, 1)) / 4.0)
    sig = float(np.sum(f(g_full[None])[0] ** 2)) - v
    return {"noise_var": v, "signal_sq": sig, "sigma": (v / sig) if sig > 0 else float("inf")}


# ------------------------------------------------------------------ subspaces (identical to the simulations)
def orth(m: np.ndarray) -> np.ndarray:
    return np.linalg.qr(m)[0]


def orth_basis(M: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis (h x r) of the column span of M, rank-revealing (SVD)."""
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, s > tol * max(s.max(), 1e-300)]


def top_k(g: np.ndarray, k: int = 2) -> np.ndarray:
    return np.linalg.svd(g, full_matrices=False)[0][:, :k]


def captured(u: np.ndarray, g: np.ndarray) -> float:
    """Mean over columns of g of |u u^T g|^2 / |g|^2 (u orthonormal)."""
    return float(np.mean(np.sum((u.T @ g) ** 2, axis=0) / np.sum(g ** 2, axis=0)))


def agreement(u: np.ndarray, v: np.ndarray) -> float:
    """Mean squared cosine of the principal angles."""
    return float(np.mean(np.linalg.svd(u.T @ v, compute_uv=False) ** 2))


def unit(x):
    return x / np.linalg.norm(x)


# ------------------------------------------------------------------ variance-matched random directions (P2)
def fast_var_matched(rng, evals, evecs, n_rows, target_var, n):
    """Copied from P2 s1d_word_specificity.fast_var_matched: r = unit(Sigma^{beta/2} z), beta by bisection so that the
    variance of the reference activations along r equals target_var (upper bracket doubled until reached)."""
    lam = np.clip(evals, 1e-12, None)
    lam_var = evals * (n_rows - 1) / n_rows
    out = []
    for _ in range(n):
        y = evecs.T @ rng.standard_normal(len(evals))

        def var_at(beta):
            c = lam ** (beta / 2) * y
            return float(np.sum(lam_var * c ** 2) / np.sum(c ** 2))

        lo, hi = 0.0, 4.0
        while var_at(hi) < target_var and hi < 64:
            hi *= 2
        for _ in range(50):
            beta = (lo + hi) / 2
            if var_at(beta) < target_var:
                lo = beta
            else:
                hi = beta
        out.append(unit(evecs @ (lam ** (beta / 2) * y)))
    return out
