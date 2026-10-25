"""Stage 1h: separate a stricter judgment from a bias toward negative-sounding tokens.

B2x (readout_B2x.csv) keeps the B2 scenarios, frames, option positions and target (the strict option), but words the
strict option in neutral/positive terms and the lenient option in negative terms. A stricter judgment raises
Delta(B2x); a negative-token bias lowers it.
Directions are rebuilt exactly as in Stage 1g (same texts, layer, alpha, eigenbasis) and checked against the saved
Stage 1g directions (cos). Nulls: 1000 random directions matched to each direction's variance, saved in order.
Manipulation checks (no steering):
  valence: log-odds of " negative" vs " positive" after 'The phrase "<option>" sounds', per option phrase;
           B2x needs lenient more negative than strict, B2 the reverse.
  judgment: item-wise correlation of baseline B2 and B2x scores after removing each option pair's mean
           (the pair prior), i.e. the scenario-driven strictness agrees across wordings.
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
from s1c_word_vs_situation import Readout
from s1d_word_specificity import fast_var_matched

SETS_1D = ("anger", "fear", "joy", "eval")
SETS_1F = ("poseval", "rare")
SEED_BASE = {"pc1_neg": 7701, "word_anger": 7702, "word_eval": 7703, "word_poseval": 7704, "pos": 7705}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1g", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", default="pc1_neg,word_anger,word_eval,word_poseval")
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED_BASE, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1h_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

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
    R_B2 = [x for x in load_csv(S / "readout_B.csv") if x["readout"] == "B2"]
    R_B2x = load_csv(S / "readout_B2x.csv")
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

    j1g = json.load(open(Path(a.res1g) / f"s1g_{tag}.json"))
    saved = dict(zip(j1g["dir_names"], np.load(Path(a.res1g) / f"s1g_dirs_{tag}.npy")))
    log.update({"alpha": alpha, "alpha_1g": j1g["alpha"], "cos_with_1g": {n: float(D[n] @ saved[n]) for n in dirs},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    save()

    # manipulation check 1: valence of each option phrase
    phrases = sorted({x[k] for x in R_B2 + R_B2x for k in ("option_a", "option_b")})
    val_items = [{"prompt": f'The phrase "{p.strip()}" sounds', "option_a": " negative", "option_b": " positive",
                  "target_option": "a"} for p in phrases]
    neg = dict(zip(phrases, Readout(tok, val_items).score(model, steer, None).tolist()))

    def strict_lenient(x):
        return (x["option_a"], x["option_b"]) if x["target_option"] == "a" else (x["option_b"], x["option_a"])

    def val_gap(rows):  # mean over distinct pairs of negativity(lenient) - negativity(strict)
        pairs = sorted({strict_lenient(x) for x in rows})
        return {f"{s.strip()}|{l.strip()}": neg[l] - neg[s] for s, l in pairs}

    gB2, gB2x = val_gap(R_B2), val_gap(R_B2x)
    log["valence_check"] = {"negativity": neg, "gap_B2": gB2, "gap_B2x": gB2x,
                            "mean_gap_B2": float(np.mean(list(gB2.values()))),
                            "mean_gap_B2x": float(np.mean(list(gB2x.values())))}
    save()

    def opt_text(x, target):
        pick_a = (x["target_option"] == "a") == target
        return x["prompt"] + (x["option_a"] if pick_a else x["option_b"])

    base_B2 = Readout(tok, R_B2).score(model, steer, None)
    Rall = Readout(tok, R_B2x)
    odd, even = R_B2x[0::2], R_B2x[1::2]
    Reven = Readout(tok, even)
    base_all, base_even = Rall.score(model, steer, None), Reven.score(model, steer, None)

    # manipulation check 2: scenario-driven strictness agrees across wordings (pair prior removed)
    pair = np.array([x["pair"] for x in R_B2x])
    dm = lambda v: v - np.array([v[pair == p].mean() for p in pair])
    log["judgment_check"] = {"corr_raw": float(np.corrcoef(base_B2, base_all)[0, 1]),
                             "corr_within_pair": float(np.corrcoef(dm(base_B2), dm(base_all))[0, 1]),
                             "base_items_B2": base_B2.tolist(), "base_items_B2x": base_all.tolist()}

    def delta(v, R=Rall, base=base_all):
        return float(np.mean(R.score(model, steer, alpha * v) - base))

    with torch.no_grad():
        d_pos = unit(last_hidden(model, tok, [opt_text(x, True) for x in odd], L).mean(0)
                     - last_hidden(model, tok, [opt_text(x, False) for x in odd], L).mean(0))
    res = log.setdefault("readout", {}).setdefault("B2x", {})
    res["base_mean"] = float(base_all.mean())
    res["delta"] = {n: delta(D[n]) for n in dirs}
    res["pos_even"] = {"delta": delta(d_pos, Reven, base_even)}
    save()
    jobs = [(n, D[n], Rall, base_all, a.n_null) for n in dirs] + [("pos", d_pos, Reven, base_even, 200)]
    for n, d, R, base, nn in jobs:
        tv = float(np.var(Ac @ d))
        rs = fast_var_matched(np.random.default_rng(SEED_BASE[n]), evals, evecs, len(Ac), tv, nn)
        z = np.asarray([delta(v, R, base) for v in rs])
        x = res["pos_even"]["delta"] if n == "pos" else res["delta"][n]
        res.setdefault("null", {})[n] = {
            "n": nn, "seed": SEED_BASE[n], "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv),
            "mean": float(z.mean()), "sd": float(z.std(ddof=1)), "q95": float(np.quantile(z, 0.95)),
            "p_one": float((np.sum(z >= x) + 1) / (nn + 1)), "p_one_lower": float((np.sum(z <= x) + 1) / (nn + 1))}
        np.save(out / f"s1h_null_B2x_{n}_{tag}.npy", z)
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("cos_with_1g", "valence_check", "readout")}, indent=1, default=float))


if __name__ == "__main__":
    main()
