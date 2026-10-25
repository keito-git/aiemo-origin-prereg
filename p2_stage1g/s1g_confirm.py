"""Stage 1g: confirmation in new models, a negated sanction readout, and the shared negative component.

Anger situations of Stage 1c; closings C + anger/fear/joy/eval (stage1d_closings.json) + poseval/rare
(stage1f_closings.json). Eigenbasis and alpha from the Stage 1d texts only (as in Stage 1f).
Directions (difference of means, last token, layer round(0.33 x layers), unit norm):
  word_X for X in anger, fear, joy, eval, poseval, rare; sit (as in 1d)
  pc1_neg   = first principal axis of {word_anger, word_fear, word_eval} (sign: positive projection on their mean)
  anger_res = word_anger with pc1_neg removed
Readouts: B2 (readout_B.csv) and B2neg (readout_B2neg.csv: the frame negated, target = the lenient option, so a
stricter judgment raises Delta in both). Positive control per readout (odd items -> even items, 200 nulls).
Manipulation check for B2neg: item-wise correlation between baseline B2 and B2neg scores (same scenario and options).
Each direction x readout: 1000 random directions matched to its own variance, saved in order.
--dirs_B2 / --dirs_B2neg select the directions to score (the Stage 1c-1f models skip what they already have).
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

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, nll_per_token, unit
from s1c_word_vs_situation import Readout
from s1d_word_specificity import fast_var_matched

SETS_1D = ("anger", "fear", "joy", "eval")
SETS_1F = ("poseval", "rare")
SEED_BASE = {"word_anger": 7501, "word_fear": 7502, "word_joy": 7503, "word_eval": 7504, "word_poseval": 7505,
             "word_rare": 7506, "sit": 7507, "pc1_neg": 7508, "anger_res": 7509, "pos": 7510}
SEED_READOUT_OFFSET = {"B2": 0, "B2neg": 100}
SEED_SPLIT = 7511


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs_B2", default="word_anger,sit,word_eval,word_poseval,word_joy,word_rare,pc1_neg,anger_res")
    ap.add_argument("--dirs_B2neg", default="word_anger,word_eval,word_poseval,pc1_neg")
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": {"base": SEED_BASE, "readout_offset": SEED_READOUT_OFFSET, "split": SEED_SPLIT},
           "t0": time.time(), "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1g_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

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
    C_dev = [r["text"] for r in ctx if r["split"] == "dev"]
    R_items = {"B2": [x for x in load_csv(S / "readout_B.csv") if x["readout"] == "B2"],
               "B2neg": load_csv(S / "readout_B2neg.csv")}
    dirs_for = {"B2": [d for d in a.dirs_B2.split(",") if d], "B2neg": [d for d in a.dirs_B2neg.split(",") if d]}

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
    np.savez_compressed(out / f"s1g_acts_{tag}.npz", ks=ks, **{k: v.astype(np.float16) for k, v in H.items()})
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + SETS_1D])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    def dirs_from(idx):
        m = {k: v[idx].mean(0) for k, v in H.items()}
        d = {f"word_{x}": unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS_1D + SETS_1F}
        d["sit"] = unit((m["Sanger"] + m["SC"]) / 2 - (m["Nanger"] + m["NC"]) / 2)
        W = np.stack([d["word_anger"], d["word_fear"], d["word_eval"]])
        pc = np.linalg.svd(W, full_matrices=False)[2][0]
        pc = pc if pc @ W.mean(0) > 0 else -pc
        d["pc1_neg"] = unit(pc)
        d["anger_res"] = unit(d["word_anger"] - (d["word_anger"] @ d["pc1_neg"]) * d["pc1_neg"])
        return d

    D = dirs_from(np.arange(len(items)))
    names = list(D)
    np.save(out / f"s1g_dirs_{tag}.npy", np.stack([D[n] for n in names]))
    perm = np.random.default_rng(SEED_SPLIT).permutation(len(items))
    D1, D2 = dirs_from(perm[: len(perm) // 2]), dirs_from(perm[len(perm) // 2:])
    base_nll = nll_per_token(model, tok, C_dev, steer, None)
    log.update({"dir_names": names, "alpha": alpha, "var": {n: float(np.var(Ac @ D[n])) for n in names},
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(names) for q in names[i + 1:]},
                "split_half_cos": {n: float(D1[n] @ D2[n]) for n in names},
                "pc1_share": float(np.linalg.svd(np.stack([D["word_anger"], D["word_fear"], D["word_eval"]]),
                                                 compute_uv=False)[0] ** 2 / 3),
                "nll_rise_dev": {n: nll_per_token(model, tok, C_dev, steer, alpha * D[n]) - base_nll for n in names}})
    save()

    def opt_text(x, target):
        pick_a = (x["target_option"] == "a") == target
        return x["prompt"] + (x["option_a"] if pick_a else x["option_b"])

    bases = {}
    for r in ("B2", "B2neg"):
        Rall = Readout(tok, R_items[r])
        odd, even = R_items[r][0::2], R_items[r][1::2]
        Reven = Readout(tok, even)
        base_all, base_even = Rall.score(model, steer, None), Reven.score(model, steer, None)
        bases[r] = base_all

        def delta(v, R=Rall, base=base_all):
            return float(np.mean(R.score(model, steer, alpha * v) - base))

        with torch.no_grad():
            d_pos = unit(last_hidden(model, tok, [opt_text(x, True) for x in odd], L).mean(0)
                         - last_hidden(model, tok, [opt_text(x, False) for x in odd], L).mean(0))
        res = log.setdefault("readout", {}).setdefault(r, {})
        res["base_mean"] = float(base_all.mean())
        res["delta"] = {n: delta(D[n]) for n in dirs_for[r]}
        res["delta_half"] = {n: [delta(D1[n]), delta(D2[n])] for n in dirs_for[r]}
        res["pos_even"] = {"delta": delta(d_pos, Reven, base_even)}
        if r == "B2neg":  # manipulation check: both scores measure strictness if the negation is understood
            log["b2neg_check"] = {"corr_base_B2_B2neg": float(np.corrcoef(bases["B2"], bases["B2neg"])[0, 1])}
        save()
        jobs = [(n, D[n], Rall, base_all, a.n_null) for n in dirs_for[r]] + [("pos", d_pos, Reven, base_even, 200)]
        for n, d, R, base, nn in jobs:
            seed = SEED_BASE[n] + SEED_READOUT_OFFSET[r]
            tv = float(np.var(Ac @ d))
            rs = fast_var_matched(np.random.default_rng(seed), evals, evecs, len(Ac), tv, nn)
            z = np.asarray([delta(v, R, base) for v in rs])
            x = res["pos_even"]["delta"] if n == "pos" else res["delta"][n]
            res.setdefault("null", {})[n] = {
                "n": nn, "seed": seed, "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv),
                "mean": float(z.mean()), "sd": float(z.std(ddof=1)), "q95": float(np.quantile(z, 0.95)),
                "p_one": float((np.sum(z >= x) + 1) / (nn + 1)), "p_one_lower": float((np.sum(z <= x) + 1) / (nn + 1))}
            np.save(out / f"s1g_null_{r}_{n}_{tag}.npy", z)
            save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("b2neg_check", "pc1_share", "readout")}, indent=1, default=float))


if __name__ == "__main__":
    main()
