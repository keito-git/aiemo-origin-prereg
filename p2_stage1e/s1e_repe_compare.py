"""Stage 1e (descriptive): where do the published RepE anger directions (Zou et al. 2023) sit relative to
d_word / d_sit, and do they move the sanction readout B2?

RepE stimuli (github.com/andyzoujm/representation-engineering, data/emotions, commit 5455d8a), empty user/assistant tags:
  concept : "Consider the anger of the following scenario:\nScenario: {s}\nAnswer: "  anger scenarios vs the other five
            emotions' scenarios. The word "anger" is in every prompt, so the contrast is carried by the scenario.
  function: "Act as if you are extremely {adj}. {s}"  adj in (angry, furious, irritated) vs (pleased, calm, peaceful),
            s = all_truncated_outputs. The contrast is carried by the emotion word.
Directions use the Stage 1d rule (difference of means, last token, layer L, unit norm) instead of RepE's PCA, so that
all directions are built the same way. perp_word = the RepE direction with the span of the four d_word[X] removed.
Delta_B2 of each direction vs 1000 random directions matched to its own variance (same eigenbasis and alpha as 1d).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit
from s1c_word_vs_situation import Readout
from s1d_word_specificity import fast_var_matched

EMOS = ("happiness", "sadness", "anger", "fear", "disgust", "surprise")
ADJ = ("angry", "furious", "irritated")
ADJ_ANT = ("pleased", "calm", "peaceful")
SEED_BASE = {"repe_concept": 7301, "repe_function": 7302, "repe_concept_perp_word": 7303, "repe_function_perp_word": 7304}
SEED_STIM = 7300


def repe_texts(repe_dir: Path):
    rng = random.Random(SEED_STIM)
    raw = {e: sorted(set(json.load(open(repe_dir / f"{e}.json")))) for e in EMOS}
    tmpl = "Consider the anger of the following scenario:\nScenario: {s}\nAnswer: "
    pos = [tmpl.format(s=s) for s in raw["anger"]]
    others = [s for e in EMOS if e != "anger" for s in raw[e]]
    neg = [tmpl.format(s=s) for s in rng.sample(others, len(pos))]
    outs = [s for s in json.load(open(repe_dir / "all_truncated_outputs.json"))]
    f_pos = [f"Act as if you are extremely {rng.choice(ADJ)}. {s}".rstrip() for s in outs]
    f_neg = [f"Act as if you are extremely {rng.choice(ADJ_ANT)}. {s}".rstrip() for s in outs]
    return {"concept": (pos, neg), "function": (f_pos, f_neg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--repe", required=True, help="representation-engineering/data/emotions")
    ap.add_argument("--s1d", required=True, help="Stage 1d results dir (anger situation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_d{a.depth}"
    log = {"args": vars(a), "seeds": {"base": SEED_BASE, "stim": SEED_STIM}, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1e_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    # Stage 1d directions and the activations that define its eigenbasis (rebuilt exactly as in s1d, anger situation)
    j1d = json.load(open(Path(a.s1d) / f"s1d_{a.model.split('/')[-1]}_anger_d{a.depth}.json"))
    D1d = dict(zip(j1d["dir_names"], np.load(Path(a.s1d) / f"s1d_dirs_{a.model.split('/')[-1]}_anger_d{a.depth}.npy")))
    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = json.load(open(S / "stage1d_closings.json"))
    K = len(cl["neutral_C"])
    texts = []
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.append(f"{it[sit]} {cl['neutral_C'][i % K]}")
            texts += [f"{it[sit]} {cl[f'{x}_L'][i % K]}" for x in ("anger", "fear", "joy", "eval")]
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    R_items = [x for x in load_csv(S / "readout_B.csv") if x["readout"] == "B2"]

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    assert L == j1d["layer"]
    steer = Steer(decoder_layers(model)[L - 1])
    with torch.no_grad():
        H_ctx = last_hidden(model, tok, C_ctx, L)
        H_1d = last_hidden(model, tok, texts, L)
        rt = repe_texts(Path(a.repe))
        D = {}
        for k, (p, n) in rt.items():
            D[f"repe_{k}"] = unit(last_hidden(model, tok, p, L).mean(0) - last_hidden(model, tok, n, L).mean(0))
    mean_norm = float(np.linalg.norm(H_ctx, axis=1).mean())
    alpha = a.alpha_frac * mean_norm
    log["alpha_check"] = {"alpha": alpha, "alpha_1d": j1d["alpha"]}
    Ac = np.concatenate([H_ctx, H_1d])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    W = np.stack([D1d[f"word_{x}"] for x in ("anger", "fear", "joy", "eval")])
    Q, _ = np.linalg.qr(W.T)
    for k in ("concept", "function"):
        v = D[f"repe_{k}"]
        D[f"repe_{k}_perp_word"] = unit(v - Q @ (Q.T @ v))
    names = list(D)
    np.save(out / f"s1e_dirs_{tag}.npy", np.stack([D[n] for n in names]))
    log.update({"layer": L, "dir_names": names, "n_stim": {k: [len(p), len(n)] for k, (p, n) in rt.items()},
                "var": {n: float(np.var(Ac @ D[n])) for n in names},
                "cos_to_1d": {n: {m: float(D[n] @ D1d[m]) for m in D1d} for n in names},
                "cos_repe": float(D["repe_concept"] @ D["repe_function"]),
                "frac_in_word_span": {k: float(np.linalg.norm(Q.T @ D[f"repe_{k}"]) ** 2) for k in ("concept", "function")}})
    save()

    R = Readout(tok, R_items)
    base = R.score(model, steer, None)

    def delta(v):
        return float(np.mean(R.score(model, steer, alpha * v) - base))

    res = log.setdefault("B2", {"delta": {n: delta(D[n]) for n in names}, "null": {}})
    save()
    for n in names:
        tv = float(np.var(Ac @ D[n]))
        rs = fast_var_matched(np.random.default_rng(SEED_BASE[n]), evals, evecs, len(Ac), tv, a.n_null)
        z = np.asarray([delta(v) for v in rs])
        x = res["delta"][n]
        res["null"][n] = {"n": a.n_null, "seed": SEED_BASE[n], "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv),
                          "mean": float(z.mean()), "sd": float(z.std(ddof=1)), "q95": float(np.quantile(z, 0.95)),
                          "p_one": float((np.sum(z >= x) + 1) / (a.n_null + 1)),
                          "p_one_lower": float((np.sum(z <= x) + 1) / (a.n_null + 1))}
        np.save(out / f"s1e_null_B2_{n}_{tag}.npy", z)
        save()
    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log[k] for k in ("cos_to_1d", "frac_in_word_span", "B2")}, indent=1, default=float))


if __name__ == "__main__":
    main()
