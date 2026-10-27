"""Stage 1k: the same-text control. 60 new scenarios (readout_R9k.csv, build_r9k.py) x {mild, severe}, each rated on
S (deserved sanction, 1-9) and C (indoors 1 - outdoors 9, designed location) with the same scenario text.
Directions, layer, alpha and the null covariance are rebuilt exactly as in Stage 1j (cos check with Stage 1g).
Two metrics per item: E = expected digit; slope = OLS slope over k of the change in log p(k) (steered minus baseline),
which is not affected by saturation at the ends of the scale (C ratings sit near 1 or 9).
Each null direction is scored on all items in one pass; the saved null array has, per draw, the columns
  dE_S, dE_C, dslope_S, dslope_C, dE_S_sev_minus_mild, dslope_S_sev_minus_mild   (index-paired across columns).
Checks (no steering): S severity effect and fraction positive (60 pairs); C severity effect; C Spearman with location.
Positive controls (direction from odd scenarios, measured on even S items, 200 nulls each):
  pos_82 = hidden(S prompt + " 8") - hidden(S prompt + " 2"); pos_sev = hidden(S severe) - hidden(S mild).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit
from s1d_word_specificity import fast_var_matched
from s1j_rating import Rating

SETS_1D = ("anger", "fear", "joy", "eval")
SETS_1F = ("poseval", "rare")
SEED_BASE = {"pc1_neg": 8201, "word_anger": 8202, "word_eval": 8203, "word_poseval": 8204, "val_eval": 8205,
             "pos_82": 8206, "pos_sev": 8207}
K9 = np.arange(1, 10)
COLS = ("dE_S", "dE_C", "dslope_S", "dslope_C", "dE_S_sev_minus_mild", "dslope_S_sev_minus_mild")


def slope(p_new: np.ndarray, p_old: np.ndarray) -> np.ndarray:
    """Per item OLS slope over k of log p_new(k) - log p_old(k)."""
    d = np.log(np.clip(p_new, 1e-30, None)) - np.log(np.clip(p_old, 1e-30, None))
    kc = K9 - K9.mean()
    return (d - d.mean(1, keepdims=True)) @ kc / (kc @ kc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1g", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", default="pc1_neg,word_anger,word_eval,word_poseval,val_eval")
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--n_null_minor", type=int, default=500, help="nulls for all directions except pc1_neg")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED_BASE, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1k_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = json.load(open(S / "stage1d_closings.json")) | json.load(open(S / "stage1f_closings.json"))
    K = len(cl["neutral_C"])
    ks = np.array([i % K for i in range(len(items))])
    texts = {}
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.setdefault(f"{sit}C", []).append(f"{it[sit]} {cl['neutral_C'][ks[i]]}")
            for x in SETS_1D + SETS_1F:
                texts.setdefault(f"{sit}{x}", []).append(f"{it[sit]} {cl[f'{x}_L'][ks[i]]}")
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    R9 = load_csv(S / "readout_R9k.csv")
    dirs = [d for d in a.dirs.split(",") if d]

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    log["layer"] = L
    steer = Steer(decoder_layers(model)[L - 1])

    with torch.no_grad():
        H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
        H_ctx = last_hidden(model, tok, C_ctx, L)
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + SETS_1D])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    m = {k: v.mean(0) for k, v in H.items()}
    D = {f"word_{x}": unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS_1D + SETS_1F}
    W = np.stack([D["word_anger"], D["word_fear"], D["word_eval"]])
    pc = np.linalg.svd(W, full_matrices=False)[2][0]
    D["pc1_neg"] = unit(pc if pc @ W.mean(0) > 0 else -pc)
    D["val_eval"] = unit(D["word_eval"] - D["word_poseval"])

    j1g = json.load(open(Path(a.res1g) / f"s1g_{tag}.json"))
    saved = dict(zip(j1g["dir_names"], np.load(Path(a.res1g) / f"s1g_dirs_{tag}.npy")))
    log.update({"alpha": alpha, "alpha_1g": j1g["alpha"],
                "cos_with_1g": {n: float(D[n] @ saved[n]) for n in dirs if n in saved},
                "cos_derived": {"val_eval|pc1_neg": float(D["val_eval"] @ D["pc1_neg"])},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    save()

    # readouts: S mild, S severe, C mild, C severe (aligned by scenario), qa format
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R9 if x["task"] == t and x["sev"] == s]
    Sm, Ss, Cm, Cs = get("S", "mild"), get("S", "severe"), get("C", "mild"), get("C", "severe")
    assert [x["scen"] for x in Sm] == [x["scen"] for x in Ss] == [x["scen"] for x in Cm] == [x["scen"] for x in Cs]
    n = len(Sm)
    loc = np.array([int(x["loc"]) for x in Sm])
    Rall = Rating(tok, Sm + Ss + Cm + Cs)
    P0 = Rall.probs(model, steer, None)
    E0 = P0 @ K9
    eSm, eSs, eCm, eCs = (E0[i * n:(i + 1) * n] for i in range(4))
    rho = lambda x, y: float(spearmanr(x, y)[0])
    eC = (eCm + eCs) / 2
    log["checks"] = {
        "severity_effect": float((eSs - eSm).mean()), "severity_frac_pos": float(((eSs - eSm) > 0).mean()),
        "C_severity_effect": float((eCs - eCm).mean()), "C_rho_location": rho(eC, loc),
        "S_base_mild": float(eSm.mean()), "S_base_severe": float(eSs.mean()), "C_base_mean": float(eC.mean()),
        "base_E": E0.tolist(), "mean_digit_probs_S": P0[:2 * n].mean(0).tolist(),
        "mean_digit_probs_C": P0[2 * n:].mean(0).tolist()}
    save()

    def effects(v) -> np.ndarray:
        P = Rall.probs(model, steer, alpha * v)
        dE, sl = (P - P0) @ K9, slope(P, P0)
        S, C = slice(0, 2 * n), slice(2 * n, 4 * n)
        return np.array([dE[S].mean(), dE[C].mean(), sl[S].mean(), sl[C].mean(),
                         dE[n:2 * n].mean() - dE[:n].mean(), sl[n:2 * n].mean() - sl[:n].mean()])

    # positive controls: direction from odd scenarios, measured on even S items
    odd = np.arange(n) % 2 == 0
    oS = lambda blk: [x["prompt"] for i, x in enumerate(blk) if odd[i]]
    with torch.no_grad():
        d_pos = {"pos_82": unit(last_hidden(model, tok, [p + " 8" for p in oS(Sm) + oS(Ss)], L).mean(0)
                                - last_hidden(model, tok, [p + " 2" for p in oS(Sm) + oS(Ss)], L).mean(0)),
                 "pos_sev": unit(last_hidden(model, tok, oS(Ss), L).mean(0) - last_hidden(model, tok, oS(Sm), L).mean(0))}
    even = [x for i, x in enumerate(Sm) if not odd[i]] + [x for i, x in enumerate(Ss) if not odd[i]]
    Reven = Rating(tok, even)
    base_even = Reven.score(model, steer, None)

    def delta_pos(v):
        return float(np.mean(Reven.score(model, steer, alpha * v) - base_even))

    res = log.setdefault("readout", {}).setdefault("R9k", {})
    res["effects"] = {nm: dict(zip(COLS, effects(D[nm]).tolist())) for nm in dirs}
    res["pos_effects_all_items"] = {k: dict(zip(COLS, effects(v).tolist())) for k, v in d_pos.items()}
    res["pos_even"] = {k: delta_pos(v) for k, v in d_pos.items()}
    res["cos_pos_82_sev"] = float(d_pos["pos_82"] @ d_pos["pos_sev"])
    save()

    for nm in dirs + list(d_pos):
        d = d_pos[nm] if nm in d_pos else D[nm]
        nn = 200 if nm in d_pos else (a.n_null if nm == "pc1_neg" else a.n_null_minor)
        tv = float(np.var(Ac @ d))
        rs = fast_var_matched(np.random.default_rng(SEED_BASE[nm]), evals, evecs, len(Ac), tv, nn)
        entry = {"n": nn, "seed": SEED_BASE[nm], "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv)}
        if nm in d_pos:
            z = np.asarray([delta_pos(v) for v in rs])
            x = res["pos_even"][nm]
            entry.update({"p_one": float((np.sum(z >= x) + 1) / (nn + 1)), "mean": float(z.mean())})
        else:
            z = np.stack([effects(v) for v in rs])
            entry["cols"] = {c: {"mean": float(z[:, j].mean()), "sd": float(z[:, j].std(ddof=1)),
                                 "p_one": float((np.sum(z[:, j] >= res["effects"][nm][c]) + 1) / (nn + 1))}
                             for j, c in enumerate(COLS)}
        np.save(out / f"s1k_null_{nm}_{tag}.npy", z)
        res.setdefault("null", {})[nm] = entry
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("cos_with_1g", "cos_derived", "readout")}, indent=1, default=float))
    print(json.dumps({k: v for k, v in log["checks"].items() if not k.startswith(("base_E", "mean_digit"))}))


if __name__ == "__main__":
    main()
