"""Stage 1d: does the behavioural effect of the emotion-word direction depend on the meaning of the word?
(design: 02_設計/04_段階1d_語の特異性と交差条件_設計_v1_20261023.md)

Situations: Stage 1c items of one emotion (S_i inducing, N_i neutral). Closings: sets X in {anger, fear, joy, eval}
plus the neutral set C (stage1d_closings.json; sentence k of every set has the word count of C_k; k = i mod 12).
  d_word[X] = mean[(S+X),(N+X)] - mean[(S+C),(N+C)]
  d_sit     = mean[(S+O),(S+C)] - mean[(N+O),(N+C)],  O = closing set of the situation's own emotion
For each readout R and each direction d: Delta_R(d) and Delta_R of n_null random directions matched to the variance
of d (seed fixed per direction and readout). Null draws are saved in order, so differences between directions and
sums over models can be tested against the paired null (index i with index i).
Robustness: bootstrap over the 12 closing clusters (directions re-estimated in each resample).
Activations at layer L (last token) are saved for later analyses.
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


def fast_var_matched(rng, evals, evecs, n_rows, target_var, n):
    """As s1c_word_vs_situation.fast_var_matched, but the upper end of the beta bracket is doubled until it
    reaches the target variance (in Stage 1c the fixed bracket [0, 4] could cap high-variance targets)."""
    lam = np.clip(evals, 1e-12, None)
    lam_var = evals * (n_rows - 1) / n_rows
    out = []
    for _ in range(n):
        y = evecs.T @ rng.standard_normal(len(evals))

        def var_at(beta):
            c = lam ** (beta / 2) * y
            return float(np.sum(lam_var * c ** 2) / np.sum(c ** 2))

        lo, hi = 0.0, 4.0
        while var_at(hi) < target_var and hi < 64:
            hi *= 2
        for _ in range(50):
            beta = (lo + hi) / 2
            if var_at(beta) < target_var:
                lo = beta
            else:
                hi = beta
        out.append(unit(evecs @ (lam ** (beta / 2) * y)))
    return out

SETS = ("anger", "fear", "joy", "eval")
SEED_BASE = {"word_anger": 7201, "word_fear": 7202, "word_joy": 7203, "word_eval": 7204, "sit": 7205, "pos": 7206}
SEED_READOUT_OFFSET = {"B1": 200, "B2": 0, "B3": 100, "B4": 300}
SEED_BOOT, SEED_SPLIT = 7207, 7208


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--situation", required=True, choices=("anger", "fear"))
    ap.add_argument("--closing_sets", default="anger,fear,joy,eval", help="closing sets X for d_word[X]")
    ap.add_argument("--readouts", default="B2", help="comma list; directions scored on every readout")
    ap.add_argument("--dirs_per_readout", default="", help="optional 'B3:word_anger,sit;...' to restrict directions")
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--boot_pair", default="word_anger,sit", help="pair for the closing-cluster bootstrap (B2 only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_{a.situation}_d{a.depth}"
    log = {"args": vars(a), "seeds": {"base": SEED_BASE, "readout_offset": SEED_READOUT_OFFSET,
                                      "boot": SEED_BOOT, "split": SEED_SPLIT}, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}}

    def save():
        (out / f"s1d_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / f"stage1c_{a.situation}.jsonl") if x.strip()]
    cl = json.load(open(S / "stage1d_closings.json"))
    xsets = [x for x in a.closing_sets.split(",") if x]
    assert a.situation in xsets and all(x in SETS for x in xsets)
    K = len(cl["neutral_C"])
    ks = np.array([i % K for i in range(len(items))])
    texts = {}
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.setdefault(f"{sit}C", []).append(f"{it[sit]} {cl['neutral_C'][ks[i]]}")
            for x in xsets:
                texts.setdefault(f"{sit}{x}", []).append(f"{it[sit]} {cl[f'{x}_L'][ks[i]]}")
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    C_dev = [r["text"] for r in ctx if r["split"] == "dev"]
    readouts = [r for r in a.readouts.split(",") if r]
    R_items = {r: [x for x in load_csv(S / "readout_B.csv") if x["readout"] == r] for r in readouts}
    log["n_items"], log["n_readout_items"] = len(items), {r: len(v) for r, v in R_items.items()}

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    log["layer"] = L
    steer = Steer(decoder_layers(model)[L - 1])  # adds v to the output of decoder layer L == hidden_states[L]

    H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
    np.savez_compressed(out / f"s1d_acts_{tag}.npz", ks=ks, **{k: v.astype(np.float16) for k, v in H.items()})
    H_ctx = last_hidden(model, tok, C_ctx, L)
    mean_norm = float(np.linalg.norm(H_ctx, axis=1).mean())
    alpha = a.alpha_frac * mean_norm
    A_all = np.concatenate([H_ctx] + list(H.values()))
    Ac = A_all - A_all.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))
    O = a.situation

    def dirs_from(idx):
        m = {k: v[idx].mean(0) for k, v in H.items()}
        d = {f"word_{x}": unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in xsets}
        d["sit"] = unit((m[f"S{O}"] + m["SC"]) / 2 - (m[f"N{O}"] + m["NC"]) / 2)
        return d

    D = dirs_from(np.arange(len(items)))
    names = list(D)
    np.save(out / f"s1d_dirs_{tag}.npy", np.stack([D[n] for n in names]))
    log["dir_names"] = names
    log["cos"] = {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(names) for q in names[i + 1:]}
    perm = np.random.default_rng(SEED_SPLIT).permutation(len(items))
    D1, D2 = dirs_from(perm[: len(perm) // 2]), dirs_from(perm[len(perm) // 2:])
    log["split_half_cos"] = {n: float(D1[n] @ D2[n]) for n in names}
    log.update({"mean_residual_norm": mean_norm, "alpha": alpha,
                "var": {n: float(np.var(Ac @ D[n])) for n in names}})
    base_nll = nll_per_token(model, tok, C_dev, steer, None)
    log["nll_rise_dev"] = {n: nll_per_token(model, tok, C_dev, steer, alpha * D[n]) - base_nll for n in names}
    save()

    restrict = {}
    for part in [p for p in a.dirs_per_readout.split(";") if p]:
        r, ds = part.split(":")
        restrict[r] = ds.split(",")

    for r in readouts:
        Rall = Readout(tok, R_items[r])
        odd, even = R_items[r][0::2], R_items[r][1::2]
        Reven = Readout(tok, even)
        base_all, base_even = Rall.score(model, steer, None), Reven.score(model, steer, None)

        def delta(v, R=Rall, base=base_all):
            return float(np.mean(R.score(model, steer, alpha * v) - base))

        def opt_text(x, target):
            pick_a = (x["target_option"] == "a") == target
            return x["prompt"] + (x["option_a"] if pick_a else x["option_b"])

        d_pos = unit(last_hidden(model, tok, [opt_text(x, True) for x in odd], L).mean(0)
                     - last_hidden(model, tok, [opt_text(x, False) for x in odd], L).mean(0))
        res = log.setdefault("readout", {}).setdefault(r, {})
        res["delta"] = {n: delta(D[n]) for n in restrict.get(r, names)}
        res["delta_half"] = {n: [delta(D1[n]), delta(D2[n])] for n in restrict.get(r, names)}
        res["pos_even"] = {"delta": delta(d_pos, Reven, base_even), "var": float(np.var(Ac @ d_pos))}
        save()
        jobs = [(n, D[n], Rall, base_all, a.n_null) for n in restrict.get(r, names)] + [("pos", d_pos, Reven, base_even, 200)]
        for n, d, R, base, nn in jobs:
            seed = SEED_BASE["pos" if n == "pos" else n] + SEED_READOUT_OFFSET[r]
            tv = float(np.var(Ac @ d))
            rs = fast_var_matched(np.random.default_rng(seed), evals, evecs, len(Ac), tv, nn)
            z = np.asarray([delta(v, R, base) for v in rs])
            x = res["pos_even"]["delta"] if n == "pos" else res["delta"][n]
            res.setdefault("null", {})[n] = {
                "n": nn, "seed": seed, "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv),
                "mean": float(z.mean()), "sd": float(z.std(ddof=1)), "q95": float(np.quantile(z, 0.95)),
                "p_one": float((np.sum(z >= x) + 1) / (nn + 1)),
                "p_one_lower": float((np.sum(z <= x) + 1) / (nn + 1))}
            np.save(out / f"s1d_null_{r}_{n}_{tag}.npy", z)
            save()

        # robustness: bootstrap over closing clusters (directions re-estimated), B2 only
        if r == "B2" and a.n_boot > 0:
            p, q = a.boot_pair.split(",")
            rng = np.random.default_rng(SEED_BOOT)
            cluster_idx = [np.where(ks == k)[0] for k in range(K)]
            diffs = []
            for _ in range(a.n_boot):
                idx = np.concatenate([cluster_idx[k] for k in rng.choice(K, K, replace=True)])
                Db = dirs_from(idx)
                diffs.append(delta(Db[p]) - delta(Db[q]))
            diffs = np.asarray(diffs)
            np.save(out / f"s1d_clusterboot_{r}_{p}-{q}_{tag}.npy", diffs)
            res["cluster_boot"] = {"pair": [p, q], "n": a.n_boot, "mean": float(diffs.mean()),
                                   "lower95_one_sided": float(np.quantile(diffs, 0.05)),
                                   "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}
            save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log[k] for k in ("layer", "split_half_cos", "readout")}, indent=1, default=float))


if __name__ == "__main__":
    main()
