"""Stage 1n gradient decomposition (plan: 02_設計/10_勾配分解解析_登録案). Runs after the Stage 1n queue.
For each question t, g_t = d/d delta of the item mean of slope_t at delta = 0, where delta is added at every position of
the output of decoder layer L-1 (the Steer hook of Stage 1n) and slope_t is the OLS slope over k of log p(k).
Since slope_t(P, P0) = w . log p with w = (k - 5) / sum (k - 5)^2, g_t is the gradient of w . log p.
g_S is split into its projection on span{g_U_bright, g_U_fam} (shared) and the residual (res).
Per direction v (the four Stage 1n directions and the regenerated Stage 1n nulls):
  pred_t(v) = alpha v.g_t (linear prediction of dslope_t), A_res(v) = alpha v.g_S^res,
  A_shared(v) = alpha v.(g_S^shared - g_U_bright), Dhat(v) = A_res + A_shared = alpha v.(g_S - g_U_bright).
The nulls are rebuilt from the Stage 1n seeds; the rebuild is checked against the saved Stage 1n files.
Descriptive extras: per-item gradients (cos with span G_U), and measured dslope at alpha and 2 alpha for the four
directions. --fd_check n compares v.g_t with central finite differences (+-alpha/4) on n random directions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit
from s1d_word_specificity import fast_var_matched
from s1j_rating import Rating
from s1m_rating import COV_SETS, SETS, paired_var_matched_rej, slope
from s1n_rating import SEED, TASKS

K9 = np.arange(1, 10)
W9 = (K9 - K9.mean()) / ((K9 - K9.mean()) @ (K9 - K9.mean()))
G_U_MAIN = ("U_bright", "U_fam")
G_U_SENS = ("U_bright", "U_fam", "U_size")


def item_grads(model, steer, R: Rating, bs: int = 8) -> np.ndarray:
    """Per item gradient of w . log p(k) w.r.t. a vector added at every position (fp32 accumulation)."""
    w = torch.tensor(W9, device="cuda", dtype=torch.float32)
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    out = []
    for b in range(0, len(R.ids), bs):
        ids, mask = R.ids[b:b + bs], R.mask[b:b + bs]
        delta = torch.zeros((ids.shape[0], 1, d), device="cuda", dtype=torch.float32, requires_grad=True)
        steer.v = delta
        with torch.enable_grad():
            lg = model(input_ids=ids, attention_mask=mask).logits
            lg = lg[torch.arange(lg.shape[0]), R.last[b:b + bs].cuda()][:, R.digits.cuda()].float()
            obj = (torch.log_softmax(lg, -1) @ w).sum()
            obj.backward()
        out.append(delta.grad[:, 0].detach().cpu().numpy().astype(np.float64))
        steer.v = None
        del lg, obj, delta
    return np.concatenate(out)


def split(g_s: np.ndarray, G: np.ndarray):
    Q = np.linalg.qr(G.T)[0]
    sh = Q @ (Q.T @ g_s)
    return sh, g_s - sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1n", required=True, help="Stage 1n result dir (dirs, pair_cos, json); empty for a smoke run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_pair", type=int, default=600)
    ap.add_argument("--n_null_minor", type=int, default=300)
    ap.add_argument("--fd_check", type=int, default=0)
    ap.add_argument("--no_measure", action="store_true", help="skip the measured alpha / 2 alpha effects")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED, "tasks": TASKS, "g_u_main": G_U_MAIN, "g_u_sens": G_U_SENS,
           "t0": time.time(), "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1n_grad_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    # stimuli, hidden states, covariance and directions exactly as in s1n_rating.py
    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = (json.load(open(S / "stage1d_closings.json")) | json.load(open(S / "stage1f_closings.json"))
          | json.load(open(S / "stage1l_closings.json")) | json.load(open(S / "stage1m_closings.json")))
    K = len(cl["neutral_C"])
    ks = np.array([i % K for i in range(len(items))])
    texts = {}
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.setdefault(f"{sit}C", []).append(f"{it[sit]} {cl['neutral_C'][ks[i]]}")
            for x, key in SETS.items():
                texts.setdefault(f"{sit}{x}", []).append(f"{it[sit]} {cl[key][ks[i]]}")
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    R9 = load_csv(S / "readout_R9n.csv")

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    L = int(round(a.depth * cfg.num_hidden_layers))
    log["layer"] = L
    steer = Steer(decoder_layers(model)[L - 1])

    with torch.no_grad():
        H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
        H_ctx = last_hidden(model, tok, C_ctx, L)
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + COV_SETS])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))
    m = {k: v.mean(0) for k, v in H.items()}
    W = {x: unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS}

    def pc1(names):
        M = np.stack([W[x] for x in names])
        pc = np.linalg.svd(M, full_matrices=False)[2][0]
        return unit(pc if pc @ M.mean(0) > 0 else -pc)

    D = {"int_neg": unit(W["anger"] - W["anger_lo"]), "int_neu": unit(W["neu_hi"] - W["neu_lo"]),
         "int_pos": unit(W["joy"] - W["joy_lo"]), "pc1_neg": pc1(("anger", "eval", "fear"))}
    dirs = list(D)
    log["alpha"] = alpha

    # nulls rebuilt from the Stage 1n seeds
    tv1, tv2 = float(np.var(Ac @ D["int_neg"])), float(np.var(Ac @ D["int_neu"]))
    c12 = float(D["int_neg"] @ D["int_neu"])
    r1, r2, got, _ = paired_var_matched_rej(np.random.default_rng(SEED["pair"]), evals, evecs, len(Ac), tv1, tv2,
                                            c12, a.n_pair)
    nulls = {"pair_neg": np.stack(r1), "pair_neu": np.stack(r2)}
    for nm in ("int_pos", "pc1_neg"):
        nulls[nm] = np.stack(fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac),
                                              float(np.var(Ac @ D[nm])), a.n_null_minor))
    R1n = Path(a.res1n)
    if (R1n / f"s1n_dirs_{tag}.npy").exists():
        j1n = json.load(open(R1n / f"s1n_{tag}.json"))
        saved = dict(zip(j1n["dir_names"], np.load(R1n / f"s1n_dirs_{tag}.npy")))
        cos_saved = np.load(R1n / f"s1n_pair_cos_{tag}.npy")
        log["rebuild_check"] = {"alpha_abs_diff": abs(alpha - j1n["alpha"]),
                                "cos_with_1n": {k: float(D[k] @ saved[k]) for k in dirs},
                                "pair_cos_max_abs_diff": float(np.max(np.abs(got - cos_saved)))}
    save()

    # readout prompts exactly as in s1n_rating.py; items ordered task x (mild, severe) x scenario
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R9 if x["task"] == t and x["sev"] == s]
    order = [(t, s) for t in TASKS for s in ("mild", "severe")]
    blocks = {k: get(*k) for k in order}
    ref = [x["scen"] for x in blocks[("S", "mild")]]
    assert all([x["scen"] for x in b] == ref for b in blocks.values())
    n = len(ref)
    Rall = Rating(tok, sum((blocks[k] for k in order), []))
    task_idx = {t: np.r_[TASKS.index(t) * 2 * n:(TASKS.index(t) + 1) * 2 * n] for t in TASKS}

    gi = item_grads(model, steer, Rall)
    g = {t: gi[task_idx[t]].mean(0) for t in TASKS}
    np.save(out / f"s1n_grad_items_{tag}.npy", gi.astype(np.float32))
    np.save(out / f"s1n_grad_mean_{tag}.npy", np.stack([g[t] for t in TASKS]))
    log["grad_norm"] = {t: float(np.linalg.norm(g[t])) for t in TASKS}
    log["grad_cos"] = {f"{p}|{q}": float(unit(g[p]) @ unit(g[q])) for i, p in enumerate(TASKS) for q in TASKS[i + 1:]}
    save()

    def decompose(names):
        sh, res = split(g["S"], np.stack([g[t] for t in names]))
        Ubr_res = split(g["U_bright"], np.stack([g[t] for t in names]))[1]
        return sh, res, float(np.linalg.norm(Ubr_res))

    dec = {}
    for key, names in (("main", G_U_MAIN), ("sens", G_U_SENS)):
        sh, res, ubr = decompose(names)
        dec[key] = (sh, res)
        log.setdefault("decomp", {})[key] = {"res_norm_frac": float(np.linalg.norm(res) / np.linalg.norm(g["S"])),
                                             "U_bright_res_norm": ubr}
        # per-item S gradients: cos with the span of the mean U gradients (descriptive)
        Q = np.linalg.qr(np.stack([g[t] for t in names]).T)[0]
        gs = gi[task_idx["S"]]
        cs = np.linalg.norm(gs @ Q, axis=1) / np.linalg.norm(gs, axis=1)
        log["decomp"][key]["item_cos_span"] = {"mean": float(cs.mean()), "q10": float(np.quantile(cs, .1)),
                                               "q50": float(np.median(cs)), "q90": float(np.quantile(cs, .9))}

    def metrics(V: np.ndarray) -> dict:
        V = np.atleast_2d(V)
        o = {f"pred_{t}": alpha * V @ g[t] for t in TASKS}
        for key, (sh, res) in dec.items():
            o[f"A_res_{key}"] = alpha * V @ res
            o[f"A_shared_{key}"] = alpha * V @ (sh - g["U_bright"])
        o["Dhat"] = alpha * V @ (g["S"] - g["U_bright"])
        return o

    keys = list(metrics(D["int_neg"]))
    log["keys"] = keys
    log["dirs"] = {nm: {k: float(v[0]) for k, v in metrics(D[nm]).items()} for nm in dirs}
    for nm, V in nulls.items():
        mt = metrics(V)
        np.save(out / f"s1n_grad_null_{nm}_{tag}.npy", np.stack([mt[k] for k in keys], 1))
    save()

    # finite-difference check of v.g_t on random unit directions
    if a.fd_check:
        rng = np.random.default_rng(8701)
        P0 = Rall.probs(model, steer, None)
        eps, rows = alpha / 4, []
        for _ in range(a.fd_check):
            u = unit(rng.standard_normal(len(g["S"])))
            sp, sm = slope(Rall.probs(model, steer, eps * u), P0), slope(Rall.probs(model, steer, -eps * u), P0)
            for t in TASKS:
                fd = float((sp[task_idx[t]].mean() - sm[task_idx[t]].mean()) / (2 * eps))
                an = float(u @ g[t])
                rows.append({"task": t, "fd": fd, "grad": an, "rel_err": abs(fd - an) / max(abs(fd), 1e-12)})
        log["fd_check"] = {"rows": rows, "median_rel_err": float(np.median([r["rel_err"] for r in rows]))}
        save()

    # measured dslope at alpha and 2 alpha for the four directions (nonlinearity, descriptive)
    if not a.no_measure:
        P0 = Rall.probs(model, steer, None)
        meas = {}
        for nm in dirs:
            for mult in (1, 2):
                sl = slope(Rall.probs(model, steer, mult * alpha * D[nm]), P0)
                meas[f"{nm}_x{mult}"] = {t: float(sl[task_idx[t]].mean()) for t in TASKS}
        log["measured"] = meas
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("rebuild_check", "decomp", "fd_check")}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
