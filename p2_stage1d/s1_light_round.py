"""Stage 1 light first round of the H3 kill test (design v2 §1, runs on the GPU container).

One model (default Qwen3-8B), one emotion (anger), one layer (relative depth 0.5).
Steps (design numbering):
  S0-4 config check; alpha selection (per-token NLL rise <= 0.15 nat on dev contexts)
  S1-1 directions: d_anger (expressive vs neutral, last token), g_GO/NE/CE/CO (design A),
       g_AG/NV (design B); all difference of means at layer L, unit norm
  S1-2 gate 0: B profile of d_anger vs 200 variance-matched random single directions
  S1-3 R2_geo and Gram matrix
  S1-4 single-variable fits rho_i vs 50 variance-matched random single directions
  S1-5 d_eff pilot: 100 random directions (own seed), PCA of the effect signature
  S1-6 Nelder-Mead pilot (k=6, max 100 evaluations, convergence logged)
  S1-7 ceiling R2_ceil (split-half re-estimate of d_anger)
Random-direction sets use separate seeds (R_base, R_null, R_deff are disjoint; F1).
Everything is written to --out as JSON / npy; nothing is decided here.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SEEDS = {"base": 1001, "null": 2002, "deff": 3003, "gate": 4004, "split": 5005}
N_VOCAB = 32000


# ---------------------------------------------------------------- model helpers
class Steer:
    """Adds a vector to the output of one decoder layer at every position."""

    def __init__(self, layer: torch.nn.Module):
        self.v = None
        self.h = layer.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.v is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.v.to(out[0].dtype),) + tuple(out[1:])
        return out + self.v.to(out.dtype)


def decoder_layers(model):
    for path in (("model", "layers"), ("model", "language_model", "layers")):
        obj = model
        try:
            for p in path:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found")


@torch.no_grad()
def last_hidden(model, tok, texts, layer_idx, bs=16):
    """hidden_states[layer_idx] (= output of decoder layer layer_idx-1, where Steer adds v) at the
    last token of each text. One text per forward pass, so there is no padding."""
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[layer_idx]
        out.append(hs[0, -1].float().cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def next_logprobs(model, tok, texts, steer, v, vocab_idx):
    steer.v = None if v is None else torch.tensor(v, device="cuda")
    rows = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to("cuda")
        lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        rows.append(lp[vocab_idx].cpu().numpy())
    steer.v = None
    return np.stack(rows).astype(np.float32)


@torch.no_grad()
def option_logodds(model, tok, items, steer, v):
    """log p(target option) - log p(other option), summed over option tokens."""
    steer.v = None if v is None else torch.tensor(v, device="cuda")
    res = []
    for it in items:
        lps = []
        for opt in (it["option_a"], it["option_b"]):
            p_ids = tok(it["prompt"], return_tensors="pt")["input_ids"][0]
            o_ids = tok(it["prompt"] + opt, return_tensors="pt")["input_ids"][0][len(p_ids):]
            ids = torch.cat([p_ids, o_ids]).unsqueeze(0).cuda()
            lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
            n = len(p_ids)
            lps.append(sum(lp[n - 1 + j, o_ids[j]].item() for j in range(len(o_ids))))
        a, b = lps
        res.append(a - b if it["target_option"] == "a" else b - a)
    steer.v = None
    return np.array(res)


@torch.no_grad()
def nll_per_token(model, tok, texts, steer, v):
    steer.v = None if v is None else torch.tensor(v, device="cuda")
    tot, n = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt").to("cuda")
        out = model(**enc, labels=enc["input_ids"])
        k = enc["input_ids"].shape[1] - 1
        tot += out.loss.item() * k
        n += k
    steer.v = None
    return tot / n


# ---------------------------------------------------------------- geometry helpers
def unit(x):
    return x / np.linalg.norm(x)


def var_matched_random(rng, evals, evecs, A_centered, target_var, n):
    """r = normalize(Sigma^{beta/2} z), beta found by bisection so that Var(a.r) ~= target_var (+-10%)."""
    out = []
    for _ in range(n):
        z = rng.standard_normal(len(evals))
        lo, hi = 0.0, 4.0
        for _ in range(40):
            beta = (lo + hi) / 2
            r = unit(evecs @ ((np.clip(evals, 1e-12, None) ** (beta / 2)) * (evecs.T @ z)))
            var = float(np.var(A_centered @ r))
            if var < target_var:
                lo = beta
            else:
                hi = beta
        out.append(r)
    return out


def r2(pred, target):
    return float(1 - np.sum((pred - target) ** 2) / np.sum(target ** 2))


def load_csv(p):
    return list(csv.DictReader(open(p)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--stim", required=True, help="folder with the Stage-1 stimulus CSVs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"model": a.model, "revision": a.revision, "seeds": SEEDS, "t0": time.time()}

    def save():
        (out / "s1_results.json").write_text(json.dumps(log, indent=1, default=float))

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    log["S0-4_config"] = {"num_hidden_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                          "vocab_size": cfg.vocab_size}
    L = int(round(a.depth * cfg.num_hidden_layers))
    log["layer"] = L
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    steer = Steer(decoder_layers(model)[L - 1])  # output of decoder layer L == hidden_states[L]

    S = Path(a.stim)
    ctx = load_csv(S / "contexts.csv")
    C_train = [r["text"] for r in ctx if r["split"] == "train"][:128]
    C_test = [r["text"] for r in ctx if r["split"] == "test"]
    C_dev = [r["text"] for r in ctx if r["split"] == "dev"]
    tgt = load_csv(S / "target_anger.csv")
    dA = load_csv(S / "appraisal_designA.csv")
    dB = load_csv(S / "appraisal_designB.csv")
    B = load_csv(S / "readout_B.csv")
    B_items = {k: [r for r in B if r["readout"] == k] for k in ("B1", "B2", "B3", "B4")}

    # vocabulary: N_VOCAB tokens with the highest mean baseline probability on train contexts
    steer.v = None
    with torch.no_grad():
        acc = None
        for t in C_train:
            enc = tok(t, return_tensors="pt").to("cuda")
            p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
            acc = p if acc is None else acc + p
    vocab_idx = torch.topk(acc, N_VOCAB).indices
    np.save(out / "vocab_idx.npy", vocab_idx.cpu().numpy())

    # activations for covariance (contexts + appraisal stimuli), layer L, last token
    t1 = time.time()
    H_ctx = last_hidden(model, tok, C_train + C_dev, L)
    Hang = last_hidden(model, tok, [r["text"] for r in tgt if r["condition"] == "anger"], L)
    Hneu = last_hidden(model, tok, [r["text"] for r in tgt if r["condition"] == "neutral"], L)
    HA = last_hidden(model, tok, [r["text"] for r in dA], L)
    HB = last_hidden(model, tok, [r["text"] for r in dB], L)
    log["t_activations_s"] = time.time() - t1
    A_all = np.concatenate([H_ctx, Hang, Hneu, HA, HB])
    mu = A_all.mean(0)
    Ac = A_all - mu
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))
    mean_norm = float(np.linalg.norm(H_ctx, axis=1).mean())
    log["mean_residual_norm"] = mean_norm

    # S1-1 directions
    d_anger = unit(Hang.mean(0) - Hneu.mean(0))
    G = {}
    for var, H, rows in (("GO", HA, dA), ("NE", HA, dA), ("CE", HA, dA), ("CO", HA, dA),
                         ("AG", HB, dB), ("NV", HB, dB)):
        lv = np.array([int(r[var]) for r in rows])
        G[var] = unit(H[lv == 1].mean(0) - H[lv == -1].mean(0))
    gB = {}
    for var in ("GO", "NE", "CE", "CO"):
        lv = np.array([int(r[var]) for r in dB])
        gB[var] = unit(HB[lv == 1].mean(0) - HB[lv == -1].mean(0))
    log["S1-1_cos_designA_vs_B"] = {v: float(G[v] @ gB[v]) for v in gB}
    names = list(G)
    Gm = np.stack([G[v] for v in names])
    np.save(out / "directions.npy", np.concatenate([d_anger[None], Gm]))
    log["S1-3_gram"] = {f"{names[i]}-{names[j]}": float(Gm[i] @ Gm[j])
                        for i in range(6) for j in range(i + 1, 6)}
    log["cos_anger_g"] = {v: float(d_anger @ G[v]) for v in names}
    Q, _ = np.linalg.qr(Gm.T)
    log["S1-3_R2_geo"] = float(np.sum((Q.T @ d_anger) ** 2))
    save()

    # alpha selection on dev contexts: largest alpha <= 0.5*mean_norm with NLL rise <= 0.15
    base_nll = nll_per_token(model, tok, C_dev, steer, None)
    alpha = 0.5 * mean_norm
    for _ in range(12):
        rise = nll_per_token(model, tok, C_dev, steer, alpha * d_anger) - base_nll
        if rise <= 0.15:
            break
        alpha *= 0.8
    log["alpha"] = {"value": alpha, "frac_of_mean_norm": alpha / mean_norm, "nll_rise": rise}
    save()

    # effect signature helpers
    base_tr = next_logprobs(model, tok, C_train, steer, None, vocab_idx)
    base_te = next_logprobs(model, tok, C_test, steer, None, vocab_idx)

    def E(v, split):
        X = C_train if split == "train" else C_test
        b = base_tr if split == "train" else base_te
        return next_logprobs(model, tok, X, steer, v, vocab_idx) - b

    # R_base: 200 variance-matched random directions (matched to Var(a.d_anger)), own seed
    var_ang = float(np.var(Ac @ d_anger))
    t1 = time.time()
    R_base = var_matched_random(np.random.default_rng(SEEDS["base"]), evals, evecs, Ac, var_ang, 200)
    base_mean = {}
    for sp in ("train", "test"):
        acc_e = None
        for r in R_base:
            e = E(alpha * r, sp)
            acc_e = e.astype(np.float64) if acc_e is None else acc_e + e
        base_mean[sp] = (acc_e / len(R_base)).astype(np.float32)
    log["t_Rbase_s"] = time.time() - t1

    def Et(v, s):
        return E(v, s) - base_mean[s]

    T = {s: Et(alpha * d_anger, s) for s in ("train", "test")}
    np.save(out / "target_effect_test.npy", T["test"].astype(np.float16))
    save()

    # S1-2 gate 0
    B0 = {k: option_logodds(model, tok, B_items[k], steer, None) for k in B_items}
    Bd = {k: float(np.mean(option_logodds(model, tok, B_items[k], steer, alpha * d_anger) - B0[k])) for k in B_items}
    R_gate = var_matched_random(np.random.default_rng(SEEDS["gate"]), evals, evecs, Ac, var_ang, 200)
    Bnull = {k: [] for k in B_items}
    for r in R_gate:
        for k in ("B1", "B2"):
            Bnull[k].append(float(np.mean(option_logodds(model, tok, B_items[k], steer, alpha * r) - B0[k])))
    log["S1-2_gate0"] = {
        "delta_anger": Bd,
        "p_B1": (sum(x >= Bd["B1"] for x in Bnull["B1"]) + 1) / 201,
        "p_B2": (sum(x >= Bd["B2"] for x in Bnull["B2"]) + 1) / 201,
        "null_q95": {k: float(np.quantile(Bnull[k], 0.95)) for k in ("B1", "B2")},
    }
    save()

    # S1-4 single-variable fits (scalar w on train, R2 on test) vs 50 random singles (R_null seed)
    def fit_scalar(v):
        xtr = Et(alpha * v, "train").ravel()
        w = float(xtr @ T["train"].ravel() / (xtr @ xtr))
        w = float(np.clip(w, -1.5, 1.5))
        return r2(w * Et(alpha * v, "test"), T["test"]), w

    log["S1-4_single"] = {v: dict(zip(("rho", "w"), fit_scalar(G[v]))) for v in names}
    rng_null = np.random.default_rng(SEEDS["null"])
    nulls = []
    for v in names:
        var_v = float(np.var(Ac @ G[v]))
        nulls += [fit_scalar(r)[0] for r in var_matched_random(rng_null, evals, evecs, Ac, var_v, 9)]
    log["S1-4_null_single"] = {"n": len(nulls), "q95": float(np.quantile(nulls, 0.95)),
                               "mean": float(np.mean(nulls))}
    save()

    # S1-5 d_eff pilot
    R_deff = var_matched_random(np.random.default_rng(SEEDS["deff"]), evals, evecs, Ac, var_ang, 100)
    M = np.stack([Et(alpha * r, "train").ravel() for r in R_deff])
    M = M - M.mean(0)
    sv = np.clip(np.linalg.eigvalsh(M @ M.T)[::-1], 0, None)
    cum = np.cumsum(sv) / sv.sum()
    log["S1-5_d_eff_90"] = int(np.searchsorted(cum, 0.90) + 1)
    save()

    # S1-6 Nelder-Mead pilot (k=6): linear init then NM, max 100 evaluations, norm constraint
    cols = np.stack([Et(alpha * G[v], "train").ravel() for v in names], 1)
    w0, *_ = np.linalg.lstsq(cols, T["train"].ravel(), rcond=None)
    max_norm = 1.5 * alpha
    hist = []

    def loss(w):
        v = alpha * (w @ Gm)
        if np.linalg.norm(v) > max_norm:
            v = v / np.linalg.norm(v) * max_norm
        val = float(np.sum((Et(v, "train") - T["train"]) ** 2))
        hist.append(val)
        return val

    res = minimize(loss, w0, method="Nelder-Mead",
                   options={"maxfev": 100, "xatol": 1e-2, "fatol": 1e-3 * abs(loss(w0))})
    v_hat = alpha * (res.x @ Gm)
    if np.linalg.norm(v_hat) > max_norm:
        v_hat = v_hat / np.linalg.norm(v_hat) * max_norm
    log["S1-6_NM"] = {"converged": bool(res.success), "nfev": int(res.nfev), "message": str(res.message),
                      "loss_init": hist[0], "loss_final": float(res.fun),
                      "rho_linear_init_test": r2(sum(w0[i] * Et(alpha * G[n], "test")
                                                     for i, n in enumerate(names)), T["test"]),
                      "rho_NM_test": r2(Et(v_hat, "test"), T["test"]), "w": res.x.tolist()}
    save()

    # S1-7 ceiling: split anger pairs in halves, re-estimate, reproduce full d_anger effect
    rng_s = np.random.default_rng(SEEDS["split"])
    idx = rng_s.permutation(len(Hang))
    h1 = unit(Hang[idx[: len(idx) // 2]].mean(0) - Hneu[idx[: len(idx) // 2]].mean(0))
    log["S1-7_R2_ceil"] = r2(Et(alpha * h1, "test"), T["test"])
    log["S1-7_cos_half_full"] = float(h1 @ d_anger)
    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: v for k, v in log.items() if k.startswith("S1") or k in ("alpha", "layer")}, indent=1, default=float))


if __name__ == "__main__":
    main()
