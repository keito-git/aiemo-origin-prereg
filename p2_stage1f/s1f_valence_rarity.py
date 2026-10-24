"""Stage 1f: is the B2 effect of word directions carried by negative valence, by any evaluative word, or by rare words?

Anger situations of Stage 1c, the Stage 1d closings (C, anger, fear, joy, eval) plus two new sets
(stage1f_closings.json, word counts equal to C_k):
  poseval: positive moral evaluation without emotion words (mirror of eval)
  rare:    low-frequency but neutral words (no evaluation, no emotion)
The eigenbasis and alpha are built from exactly the Stage 1d texts, so nulls are comparable with Stage 1d.
Directions scored on B2: word_eval (re-run, seed as in 1d), word_poseval, word_rare, and sit_scaled
(d_sit with alpha multiplied by sd(word_anger)/sd(sit), so that the push in units of its own sd equals that of
word_anger; its null uses the same scaled alpha). Also: mean surprisal of the closing tokens per set.
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
SEED_BASE = {"word_eval": 7204, "word_poseval": 7401, "word_rare": 7402, "sit_scaled": 7403}


@torch.no_grad()
def closing_surprisal(model, tok, prefixes, full):
    """Mean NLL (nats/token) of the closing tokens of each text given its situation prefix."""
    out = []
    for p, t in zip(prefixes, full):
        n_p = len(tok(p)["input_ids"])
        ids = tok(t, return_tensors="pt")["input_ids"].to("cuda")
        lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        tgt = ids[0, n_p:]
        out.append(float(-lp[n_p - 1:-1].gather(1, tgt[:, None]).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
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
        (out / f"s1f_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

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
    R_items = [x for x in load_csv(S / "readout_B.csv") if x["readout"] == "B2"]

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
    np.savez_compressed(out / f"s1f_acts_{tag}.npz", ks=ks, **{k: v.astype(np.float16) for k, v in H.items()})
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    keys_1d = [k for k in H if k[1:] in ("C",) + SETS_1D]
    Ac = np.concatenate([H_ctx] + [H[k] for k in keys_1d])  # Stage 1d texts only
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    m = {k: v.mean(0) for k, v in H.items()}
    D = {f"word_{x}": unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS_1D + SETS_1F}
    D["sit"] = unit((m["Sanger"] + m["SC"]) / 2 - (m["Nanger"] + m["NC"]) / 2)
    sd = {n: float(np.std(Ac @ D[n])) for n in D}
    scale_sit = sd["word_anger"] / sd["sit"]
    log.update({"alpha": alpha, "scale_sit": scale_sit, "var": {n: s ** 2 for n, s in sd.items()},
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(D) for q in list(D)[i + 1:]}})
    d1 = Path(a.out).parent / "p2_stage1d"
    f1d = d1 / f"s1d_{a.model.split('/')[-1]}_anger_d{a.depth}.json"
    if f1d.exists():  # reproducibility check against Stage 1d
        j1d = json.load(open(f1d))
        D1d = dict(zip(j1d["dir_names"], np.load(d1 / f"s1d_dirs_{a.model.split('/')[-1]}_anger_d{a.depth}.npy")))
        log["cos_with_1d"] = {n: float(D[n] @ D1d[n]) for n in D1d}
        log["alpha_1d"] = j1d["alpha"]

    # surprisal of the closing tokens (no steering)
    prefixes = [it[s] for it in items for s in ("S", "N")]
    log["closing_surprisal"] = {}
    for x in ("C",) + SETS_1D + SETS_1F:
        key = "neutral_C" if x == "C" else f"{x}_L"
        full = [f"{it[s]} {cl[key][ks[i]]}" for i, it in enumerate(items) for s in ("S", "N")]
        pre = [p for p in prefixes]
        log["closing_surprisal"][x] = float(np.mean(closing_surprisal(model, tok, pre, full)))
    save()

    R = Readout(tok, R_items)
    base = R.score(model, steer, None)

    def delta(v, scale=1.0):
        return float(np.mean(R.score(model, steer, scale * alpha * v) - base))

    jobs = {"word_eval": (D["word_eval"], 1.0), "word_poseval": (D["word_poseval"], 1.0),
            "word_rare": (D["word_rare"], 1.0), "sit_scaled": (D["sit"], scale_sit)}
    res = log.setdefault("B2", {"delta": {n: delta(v, s) for n, (v, s) in jobs.items()}, "null": {}})
    save()
    for n, (d, s) in jobs.items():
        tv = float(np.var(Ac @ d))
        rs = fast_var_matched(np.random.default_rng(SEED_BASE[n]), evals, evecs, len(Ac), tv, a.n_null)
        z = np.asarray([delta(v, s) for v in rs])
        x = res["delta"][n]
        res["null"][n] = {"n": a.n_null, "seed": SEED_BASE[n], "scale": s,
                          "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv),
                          "mean": float(z.mean()), "sd": float(z.std(ddof=1)), "q95": float(np.quantile(z, 0.95)),
                          "p_one": float((np.sum(z >= x) + 1) / (a.n_null + 1)),
                          "p_one_lower": float((np.sum(z <= x) + 1) / (a.n_null + 1))}
        np.save(out / f"s1f_null_B2_{n}_{tag}.npy", z)
        save()
    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log[k] for k in ("closing_surprisal", "B2")}, indent=1, default=float))


if __name__ == "__main__":
    main()
