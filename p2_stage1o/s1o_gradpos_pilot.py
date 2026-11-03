"""Stage 1o pilot 3: gradient-based held-out positive controls (design 11 §4, external review 2; 2026-11-03, before any
Stage 1o registration). Only random and positive-control directions are applied; the Stage 1n test directions are
built for their variance only and are never applied.
Ten readouts on the 60 Stage 1k scenarios x mild/severe:
  digit (1-9): S, HARSH, L, U_bright, U_fam (readout_R9n.csv)
  4-choice (four letter rotations): CAT_S, CAT_U (catprobe), CAT_R (catprobe2), CAT_S2 (catprobe3)
  2-choice (adjacent pairs, both orders): PAIR (catprobe3)
Every readout is linear in the log choice probabilities: score_i = w_i . log softmax(logits over the choice tokens),
  digit: w = (k - 5) / sum (k - 5)^2 (the Stage 1n slope); 4-choice: w at each letter = (rank - 2.5) / 5 (rslope of
  pilot 2); PAIR: +1 at the heavier letter, -1 at the lighter one (log-odds of the heavier option).
The effect of a direction is the mean over items of w_i . (log p_steered - log p_base).
Positive control of readout t: unit(mean gradient of score_t over the odd scenarios K01, K03, ...), applied at alpha
and read out on the even scenarios; compared with n_null variance-matched random directions (target variance =
var(int_neg), seed 8621) read out on the same even items (one-sided p). The full 10 x 10 matrix (each control on
every readout) is reported for selectivity.
Writes s1o_gradpos_{tag}.json and s1o_gradpos_null_{tag}.npy (n_null x readouts, even items)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit
from s1d_word_specificity import fast_var_matched
from s1m_rating import COV_SETS, SETS

SEED_NULL = 8621
DIGIT = ("S", "HARSH", "L", "U_bright", "U_fam")
CAT4 = ("CAT_S", "CAT_U", "CAT_R", "CAT_S2")
READ = DIGIT + CAT4 + ("PAIR",)
K9 = np.arange(1, 10)
W9 = (K9 - K9.mean()) / ((K9 - K9.mean()) @ (K9 - K9.mean()))


class Readout:
    """Items of one readout: token ids up to the answer position, the choice tokens and per-item weights."""

    def __init__(self, tok, items, choices: list[str], W: np.ndarray):
        seqs, ids = [], None
        for it in items:
            full = [tok(it["prompt"] + f" {k}")["input_ids"] for k in choices]
            pre = full[0][:-1]
            assert all(f[:-1] == pre for f in full), it["item_id"]
            d = [f[-1] for f in full]
            assert ids is None or d == ids
            ids = d
            seqs.append(pre)
        T = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.ids = torch.full((len(seqs), T), pad, dtype=torch.long).cuda()
        self.mask = torch.zeros((len(seqs), T), dtype=torch.long).cuda()
        for j, s in enumerate(seqs):
            self.ids[j, :len(s)] = torch.tensor(s)
            self.mask[j, :len(s)] = 1
        self.last = torch.tensor([len(s) - 1 for s in seqs]).cuda()
        self.choice = torch.tensor(ids).cuda()
        self.W = W
        self.even = np.array([int(x["scen"][1:]) % 2 == 0 for x in items])

    @torch.no_grad()
    def logp(self, model, steer, v, bs: int = 50) -> np.ndarray:
        steer.v = None if v is None else torch.tensor(v, device="cuda")
        out = []
        for b in range(0, len(self.ids), bs):
            lg = model(input_ids=self.ids[b:b + bs], attention_mask=self.mask[b:b + bs]).logits
            lg = lg[torch.arange(lg.shape[0]), self.last[b:b + bs]][:, self.choice].float()
            out.append(torch.log_softmax(lg, -1).cpu().numpy())
        steer.v = None
        return np.concatenate(out)

    def mean_grad(self, model, steer, d: int, sel: np.ndarray, bs: int = 8) -> np.ndarray:
        """Mean over the selected items of d score_i / d delta (delta added at every position; fp32)."""
        idx = np.flatnonzero(sel)
        acc = np.zeros(d)
        for b in range(0, len(idx), bs):
            j = torch.tensor(idx[b:b + bs]).cuda()
            w = torch.tensor(self.W[idx[b:b + bs]], device="cuda", dtype=torch.float32)
            delta = torch.zeros((len(j), 1, d), device="cuda", dtype=torch.float32, requires_grad=True)
            steer.v = delta
            with torch.enable_grad():
                lg = model(input_ids=self.ids[j], attention_mask=self.mask[j]).logits
                lg = lg[torch.arange(len(j)), self.last[j]][:, self.choice].float()
                obj = (torch.log_softmax(lg, -1) * w).sum()
                obj.backward()
            acc += delta.grad[:, 0].sum(0).detach().cpu().numpy().astype(np.float64)
            steer.v = None
            del lg, obj, delta
        return acc / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seed_null": SEED_NULL, "readouts": READ, "t0": time.time()}

    def save():
        (out / f"s1o_gradpos_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    # covariance and the variance of int_neg exactly as in s1o_null_pilot2.py (int_neg itself is never applied)
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

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    L = int(round(a.depth * cfg.num_hidden_layers))
    steer = Steer(decoder_layers(model)[L - 1])
    with torch.no_grad():
        H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
        H_ctx = last_hidden(model, tok, C_ctx, L)
    hdim = H_ctx.shape[1]
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + COV_SETS])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))
    m = {k: v.mean(0) for k, v in H.items()}
    W = {x: unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS}
    tv_neg = float(np.var(Ac @ unit(W["anger"] - W["anger_lo"])))
    log.update({"layer": L, "alpha": alpha, "target_var_int_neg": tv_neg})
    save()

    # readouts
    def rows(fn, task):
        R = [x for x in load_csv(S / fn) if x["task"] == task]
        for x in R:
            x["prompt"] = x["prompt_qa"] + "\nAnswer:"
        return R

    RO = {}
    for t in DIGIT:
        R = rows("readout_R9n.csv", t)
        RO[t] = Readout(tok, R, [str(k) for k in K9], np.tile(W9, (len(R), 1)))
    for t, fn in (("CAT_S", "readout_R9o_catprobe.csv"), ("CAT_U", "readout_R9o_catprobe.csv"),
                  ("CAT_R", "readout_R9o_catprobe2.csv"), ("CAT_S2", "readout_R9o_catprobe3.csv")):
        R = rows(fn, t)
        assert sorted({x["rot"] for x in R}) == ["0", "1", "2", "3"] and len(R) == 480, t
        rk = np.array([[int(c) for c in x["ranks"]] for x in R], dtype=float)
        RO[t] = Readout(tok, R, list("ABCD"), (rk - 2.5) / 5.0)
    R = rows("readout_R9o_catprobe3.csv", "PAIR")
    assert len(R) == 720
    rk = np.array([[int(c) for c in x["ranks"]] for x in R], dtype=float)
    Wp = np.where(rk == rk.max(1, keepdims=True), 1.0, -1.0)
    assert all((Wp[i, 1] == 1.0) == (x["rot"][-1] == "a") for i, x in enumerate(R))   # 'a' order: heavier at B
    RO["PAIR"] = Readout(tok, R, list("AB"), Wp)
    log["n_items"] = {t: int(len(r.W)) for t, r in RO.items()}

    base = {t: r.logp(model, steer, None) for t, r in RO.items()}

    def eff(v) -> np.ndarray:
        """Effect on every readout (even items only)."""
        o = []
        for t in READ:
            r = RO[t]
            d = ((r.logp(model, steer, alpha * v) - base[t]) * r.W).sum(1)
            o.append(d[r.even].mean())
        return np.array(o)

    # gradient-based positive controls from the odd scenarios
    t1 = time.time()
    G = {t: RO[t].mean_grad(model, steer, hdim, ~RO[t].even) for t in READ}
    dpos = {t: unit(g) for t, g in G.items()}
    log["t_grad_s"] = time.time() - t1
    log["grad_norm"] = {t: float(np.linalg.norm(g)) for t, g in G.items()}
    log["cos_grad"] = {f"{p}|{q}": float(dpos[p] @ dpos[q]) for i, p in enumerate(READ) for q in READ[i + 1:]}
    log["var_ratio_pos"] = {t: float(np.var(Ac @ v) / tv_neg) for t, v in dpos.items()}
    save()

    # null on the even items
    rs = fast_var_matched(np.random.default_rng(SEED_NULL), evals, evecs, len(Ac), tv_neg, a.n_null)
    Z = np.stack([eff(r) for r in rs])                                   # n_null x readouts
    np.save(out / f"s1o_gradpos_null_{tag}.npy", Z)
    mu, sd = Z.mean(0), Z.std(0, ddof=1)
    log["null"] = {"n": a.n_null, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv_neg),
                   "mean": dict(zip(READ, mu.tolist())), "sd": dict(zip(READ, sd.tolist()))}
    save()

    M = np.stack([eff(dpos[t]) for t in READ])                          # control x readout
    log["pos_matrix_obs"] = M.tolist()
    log["pos_matrix_z"] = ((M - mu) / sd).tolist()
    log["pos"] = {t: {"obs": float(M[i, i]), "z": float((M[i, i] - mu[i]) / sd[i]),
                      "p_one": float((np.sum(Z[:, i] >= M[i, i]) + 1) / (a.n_null + 1)),
                      "pass_05": bool((np.sum(Z[:, i] >= M[i, i]) + 1) / (a.n_null + 1) < .05)}
                  for i, t in enumerate(READ)}
    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log[k] for k in ("alpha", "var_ratio_pos", "pos")}, indent=1, default=float))


if __name__ == "__main__":
    main()
