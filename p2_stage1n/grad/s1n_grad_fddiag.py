"""Diagnostic for the s1n_grad.py smoke (unplanned model only): finite differences of the item-mean slope_t at several
step sizes, on random unit directions and on the unit S gradient, against the backprop gradient. Separates bf16 noise
(error grows as eps shrinks) from curvature (error grows with eps). Reports error relative to the RMS of the gradient
projections, which stays defined when a projection is near zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit
from s1j_rating import Rating
from s1m_rating import slope
from s1n_grad import item_grads
from s1n_rating import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_dir", type=int, default=4)
    a = ap.parse_args()
    S = Path(a.stim)
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    R9 = load_csv(S / "readout_R9n.csv")
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    L = int(round(a.depth * cfg.num_hidden_layers))
    steer = Steer(decoder_layers(model)[L - 1])
    alpha = a.alpha_frac * float(np.linalg.norm(last_hidden(model, tok, C_ctx, L), axis=1).mean())
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    order = [(t, s) for t in TASKS for s in ("mild", "severe")]
    blocks = {k: [x for x in R9 if x["task"] == k[0] and x["sev"] == k[1]] for k in order}
    n = len(blocks[("S", "mild")])
    Rall = Rating(tok, sum((blocks[k] for k in order), []))
    task_idx = {t: np.r_[TASKS.index(t) * 2 * n:(TASKS.index(t) + 1) * 2 * n] for t in TASKS}
    gi = item_grads(model, steer, Rall)
    g = {t: gi[task_idx[t]].mean(0) for t in TASKS}
    rng = np.random.default_rng(8702)
    U = [("rand%d" % i, unit(rng.standard_normal(len(g["S"])))) for i in range(a.n_dir)] + [("gS", unit(g["S"]))]
    P0 = Rall.probs(model, steer, None)
    res = {"alpha": alpha, "rows": []}
    for frac in (1 / 16, 1 / 8, 1 / 4, 1 / 2):
        eps = alpha * frac
        for nm, u in U:
            sp, sm = slope(Rall.probs(model, steer, eps * u), P0), slope(Rall.probs(model, steer, -eps * u), P0)
            for t in TASKS:
                res["rows"].append({"frac": frac, "dir": nm, "task": t, "grad": float(u @ g[t]),
                                    "fd": float((sp[task_idx[t]].mean() - sm[task_idx[t]].mean()) / (2 * eps))})
    summ = {}
    for frac in (1 / 16, 1 / 8, 1 / 4, 1 / 2):
        rr = [r for r in res["rows"] if r["frac"] == frac and r["dir"] != "gS"]
        gr, fd = np.array([r["grad"] for r in rr]), np.array([r["fd"] for r in rr])
        summ[str(frac)] = {"rms_err_over_rms_grad": float(np.sqrt(np.mean((fd - gr) ** 2)) / np.sqrt(np.mean(gr ** 2))),
                           "corr": float(np.corrcoef(gr, fd)[0, 1]),
                           "gS_on_S_ratio": [r["fd"] / r["grad"] for r in res["rows"]
                                             if r["frac"] == frac and r["dir"] == "gS" and r["task"] == "S"][0]}
    res["summary"] = summ
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "s1n_grad_fddiag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
