"""Stage 1k pre-registration probe (no target direction is steered): is the same-text control question valid, and
which positive control works?

readout_R9k.csv (build_r9k.py): 60 new scenarios x {mild, severe} x {S: deserved sanction, C: indoors/outdoors}.
Baseline checks per model:
  S: severity effect (mean E(severe) - E(mild) over 60 scenarios) and fraction positive.
  C: severity effect (should be near 0), validity = Spearman(E_C, designed location), mean |E_C(outdoor) - E_C(indoor)|.
Positive controls (direction from odd scenarios, measured on even scenarios, 50 variance-matched nulls on the context
covariance):
  pos_82:  hidden(S prompt + " 8") - hidden(S prompt + " 2")      (the Stage 1j positive control)
  pos_sev: hidden(S prompt, severe) - hidden(S prompt, mild)       (the severity manipulation itself)
For each: Delta E on even S items and on even C items, and Delta slope (OLS slope over k of the change in log p(k),
which is not affected by saturation at the ends of the scale).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit
from s1d_word_specificity import fast_var_matched
from s1j_rating import Rating

K = np.arange(1, 10)


def slope(p_new: np.ndarray, p_old: np.ndarray) -> np.ndarray:
    """Per item OLS slope over k of log p_new(k) - log p_old(k)."""
    d = np.log(np.clip(p_new, 1e-30, None)) - np.log(np.clip(p_old, 1e-30, None))
    kc = K - K.mean()
    return (d - d.mean(1, keepdims=True)) @ kc / (kc @ kc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=50)
    a = ap.parse_args()
    S = Path(a.stim)
    R = load_csv(S / "readout_R9k.csv")
    for x in R:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R if x["task"] == t and x["sev"] == s]
    Sm, Ss, Cm, Cs = get("S", "mild"), get("S", "severe"), get("C", "mild"), get("C", "severe")
    assert [x["scen"] for x in Sm] == [x["scen"] for x in Ss] == [x["scen"] for x in Cm] == [x["scen"] for x in Cs]
    n = len(Sm)
    loc = np.array([int(x["loc"]) for x in Sm])
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    steer = Steer(decoder_layers(model)[L - 1])
    H_ctx = last_hidden(model, tok, C_ctx, L)
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    Ac = H_ctx - H_ctx.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    Rall = Rating(tok, Sm + Ss + Cm + Cs)
    P0 = Rall.probs(model, steer, None)
    E0 = P0 @ K
    eSm, eSs, eCm, eCs = (E0[i * n:(i + 1) * n] for i in range(4))
    rho = lambda u, v: float(spearmanr(u, v)[0])
    eC = (eCm + eCs) / 2
    res = {"model": a.model, "layer": L, "alpha": alpha, "checks": {
        "S_severity_effect": float((eSs - eSm).mean()), "S_severity_frac_pos": float(((eSs - eSm) > 0).mean()),
        "S_base_mild": float(eSm.mean()), "S_base_severe": float(eSs.mean()), "S_sd_scen": float(np.std(eSm + eSs)),
        "C_severity_effect": float((eCs - eCm).mean()), "C_rho_location": rho(eC, loc),
        "C_outdoor_minus_indoor": float(eC[loc == 1].mean() - eC[loc == 0].mean()),
        "C_base_mean": float(eC.mean()), "S_C_rho": rho((eSm + eSs) / 2, eC),
        "mean_digit_probs_S": P0[:2 * n].mean(0).tolist(), "mean_digit_probs_C": P0[2 * n:].mean(0).tolist()}}
    print(a.model, {k: v for k, v in res["checks"].items() if not k.startswith("mean_digit")}, flush=True)

    odd = np.arange(n) % 2 == 0   # scenarios used to build the direction
    ev = ~odd                     # scenarios used to measure
    ix = lambda blk: np.concatenate([np.arange(n)[ev] + b * n for b in blk])
    iS, iC = ix((0, 1)), ix((2, 3))
    Rev = Rating(tok, [(Sm + Ss + Cm + Cs)[i] for i in np.concatenate([iS, iC])])
    P0e = P0[np.concatenate([iS, iC])]
    m = len(iS)

    def eff(v):
        P = Rev.probs(model, steer, alpha * v)
        dE = (P - P0e) @ K
        sl = slope(P, P0e)
        return np.array([dE[:m].mean(), dE[m:].mean(), sl[:m].mean(), sl[m:].mean()])

    oS = lambda blk: [x["prompt"] for i, x in enumerate(blk) if odd[i]]
    d = {"pos_82": unit(last_hidden(model, tok, [p + " 8" for p in oS(Sm) + oS(Ss)], L).mean(0)
                        - last_hidden(model, tok, [p + " 2" for p in oS(Sm) + oS(Ss)], L).mean(0)),
         "pos_sev": unit(last_hidden(model, tok, oS(Ss), L).mean(0) - last_hidden(model, tok, oS(Sm), L).mean(0))}
    res["cos_pos_82_sev"] = float(d["pos_82"] @ d["pos_sev"])
    names = ("dE_S", "dE_C", "dslope_S", "dslope_C")
    for k, v in d.items():
        obs = eff(v)
        rs = fast_var_matched(np.random.default_rng(8101 if k == "pos_82" else 8102), evals, evecs, len(Ac),
                              float(np.var(Ac @ v)), a.n_null)
        z = np.stack([eff(r) for r in rs])
        res[k] = {nm: {"obs": float(obs[j]), "null_mean": float(z[:, j].mean()), "null_sd": float(z[:, j].std(ddof=1)),
                       "p_upper": float((np.sum(z[:, j] >= obs[j]) + 1) / (len(z) + 1))} for j, nm in enumerate(names)}
        print(k, {nm: (round(r["obs"], 3), round(r["p_upper"], 3)) for nm, r in res[k].items()}, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"s1k_probe_{a.model.split('/')[-1]}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
