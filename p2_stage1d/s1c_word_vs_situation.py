"""Stage 1c: is the behavioural effect of an emotion direction carried by the emotion word or by the situation?
(design: 02_設計/03_段階1c_語彙と状況の分解_設計_v1_20261023.md)

2x2 minimal pairs per item i: situation {S_i (emotion-inducing), N_i (neutral)} x closing {L_k (one emotion
word), C_k (neutral, same word count)}, k = i mod 12. Last token = end of the closing in every version.
  d_word = mean[(S+L),(N+L)] - mean[(S+C),(N+C)]
  d_sit  = mean[(S+L),(S+C)] - mean[(N+L),(N+C)]
Primary test (preregistered): Delta_R(d_word) - Delta_R(d_sit) > 0, item bootstrap (one-sided 95% lower bound > 0).
Secondary: each direction vs 1,000 random directions matched to its own variance; split-half reliability;
positive control pos_R (odd readout items -> even items).
Delta_R = mean over readout items of the change in log p(target option) - log p(other option).
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

SEEDS = {"null_word": 7101, "null_sit": 7102, "null_pos": 7103, "boot": 7104, "split": 7105}


def fast_var_matched(rng, evals, evecs, n_rows, target_var, n):
    """Same construction as s1_light_round.var_matched_random (r = normalize(Sigma^{beta/2} z), beta by
    bisection), but Var(a.r) is computed in the eigenbasis: sum(lam_i c_i^2) / sum(c_i^2), c = lam^{beta/2} * (V^T z).
    lam is rescaled from np.cov (ddof=1) to np.var (ddof=0) so the matched quantity is identical."""
    lam = np.clip(evals, 1e-12, None)
    lam_var = evals * (n_rows - 1) / n_rows
    out = []
    for _ in range(n):
        y = evecs.T @ rng.standard_normal(len(evals))
        lo, hi = 0.0, 4.0
        for _ in range(40):
            beta = (lo + hi) / 2
            c = lam ** (beta / 2) * y
            var = float(np.sum(lam_var * c ** 2) / np.sum(c ** 2))
            if var < target_var:
                lo = beta
            else:
                hi = beta
        out.append(unit(evecs @ (lam ** (beta / 2) * y)))
    return out


def build_texts(items: list[dict], closings: dict, emotion: str) -> dict[str, list[str]]:
    L, C = closings[f"{emotion}_L"], closings["neutral_C"]
    out = {"SL": [], "SC": [], "NL": [], "NC": []}
    for i, it in enumerate(items):
        k = i % len(L)
        out["SL"].append(f"{it['S']} {L[k]}")
        out["SC"].append(f"{it['S']} {C[k]}")
        out["NL"].append(f"{it['N']} {L[k]}")
        out["NC"].append(f"{it['N']} {C[k]}")
    return out


class Readout:
    """Batched two-option log-odds. Option tokens are taken from the tokenisation of prompt+option as a whole
    (its last n tokens, n = len(tok(prompt+opt)) - len(tok(prompt))), so ids always equal tok(prompt+opt)."""

    def __init__(self, tok, items: list[dict]):
        seqs, spans = [], []
        for it in items:
            n_p = len(tok(it["prompt"])["input_ids"])
            for opt in (it["option_a"], it["option_b"]):
                ids = tok(it["prompt"] + opt)["input_ids"]
                seqs.append(ids)
                s = min(max(n_p, 1), len(ids) - 1)  # at least one option token
                spans.append((s, len(ids)))
        T = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        self.mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, s in enumerate(seqs):
            self.ids[j, : len(s)] = torch.tensor(s)
            self.mask[j, : len(s)] = 1
        self.spans = spans
        self.sign = torch.tensor([1.0 if it["target_option"] == "a" else -1.0 for it in items])
        self.ids, self.mask = self.ids.cuda(), self.mask.cuda()

    @torch.no_grad()
    def score(self, model, steer, v, bs: int = 64) -> np.ndarray:
        steer.v = None if v is None else torch.tensor(v, device="cuda")
        lps = []
        for b in range(0, len(self.ids), bs):
            lp = torch.log_softmax(model(input_ids=self.ids[b:b + bs], attention_mask=self.mask[b:b + bs]).logits.float(), -1)
            for j in range(lp.shape[0]):
                s, e = self.spans[b + j]
                tgt = self.ids[b + j, s:e]
                lps.append(lp[j, s - 1:e - 1].gather(1, tgt[:, None]).sum().item())
        steer.v = None
        lps = np.array(lps).reshape(-1, 2)
        return (lps[:, 0] - lps[:, 1]) * self.sign.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--emotion", default="anger")
    ap.add_argument("--readout", default="B2")
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--stim", required=True, help="folder with contexts.csv, readout_B.csv, stage1c_closings.json, raw_generation/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_{a.emotion}_{a.readout}_d{a.depth}"
    log = {"args": vars(a), "seeds": SEEDS, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}}

    def save():
        (out / f"s1c_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / f"stage1c_{a.emotion}.jsonl") if x.strip()]
    closings = json.load(open(S / "stage1c_closings.json"))
    texts = build_texts(items, closings, a.emotion)
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    C_dev = [r["text"] for r in ctx if r["split"] == "dev"]
    R_items = [r for r in load_csv(S / "readout_B.csv") if r["readout"] == a.readout]
    log["n_items"], log["n_readout_items"] = len(items), len(R_items)

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    log["layer"] = L
    steer = Steer(decoder_layers(model)[L - 1])  # adds v to the output of decoder layer L == hidden_states[L]

    H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
    H_ctx = last_hidden(model, tok, C_ctx, L)
    mean_norm = float(np.linalg.norm(H_ctx, axis=1).mean())
    alpha = a.alpha_frac * mean_norm
    A_all = np.concatenate([H_ctx] + list(H.values()))
    Ac = A_all - A_all.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    def dirs_from(idx):
        w = (H["SL"][idx].mean(0) + H["NL"][idx].mean(0)) / 2 - (H["SC"][idx].mean(0) + H["NC"][idx].mean(0)) / 2
        s = (H["SL"][idx].mean(0) + H["SC"][idx].mean(0)) / 2 - (H["NL"][idx].mean(0) + H["NC"][idx].mean(0)) / 2
        i_ = (H["SL"][idx].mean(0) - H["NL"][idx].mean(0)) - (H["SC"][idx].mean(0) - H["NC"][idx].mean(0))
        return unit(w), unit(s), unit(i_)

    all_idx = np.arange(len(items))
    d_word, d_sit, d_int = dirs_from(all_idx)
    np.save(out / f"s1c_dirs_{tag}.npy", np.stack([d_word, d_sit, d_int]))

    # positive control direction from odd readout items (target option vs other option, last token)
    odd, even = R_items[0::2], R_items[1::2]

    def opt_text(r, target):
        pick_a = (r["target_option"] == "a") == target
        return r["prompt"] + (r["option_a"] if pick_a else r["option_b"])

    d_pos = unit(last_hidden(model, tok, [opt_text(r, True) for r in odd], L).mean(0)
                 - last_hidden(model, tok, [opt_text(r, False) for r in odd], L).mean(0))
    var = {n: float(np.var(Ac @ d)) for n, d in (("word", d_word), ("sit", d_sit), ("pos", d_pos))}
    log.update({"mean_residual_norm": mean_norm, "alpha": alpha, "var": var,
                "cos": {"word-sit": float(d_word @ d_sit), "word-int": float(d_word @ d_int),
                        "sit-int": float(d_sit @ d_int), "pos-word": float(d_pos @ d_word), "pos-sit": float(d_pos @ d_sit)}})

    # quality: split-half reliability of each direction
    perm = np.random.default_rng(SEEDS["split"]).permutation(len(items))
    h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
    w1, s1, _ = dirs_from(h1)
    w2, s2, _ = dirs_from(h2)
    log["split_half_cos"] = {"word": float(w1 @ w2), "sit": float(s1 @ s2)}
    log["nll_rise_dev"] = {n: nll_per_token(model, tok, C_dev, steer, alpha * d) - nll_per_token(model, tok, C_dev, steer, None)
                           for n, d in (("word", d_word), ("sit", d_sit))}
    save()

    R_all, R_even = Readout(tok, R_items), Readout(tok, even)
    base_all, base_even = R_all.score(model, steer, None), R_even.score(model, steer, None)

    def delta(v, R=R_all, base=None):
        base = base_all if base is None else base
        return float(np.mean(R.score(model, steer, alpha * v) - base))

    obs = {"word": delta(d_word), "sit": delta(d_sit), "int": delta(d_int),
           "pos_even": delta(d_pos, R_even, base_even)}
    log["delta"] = obs
    log["delta_half"] = {"word_h1": delta(w1), "word_h2": delta(w2), "sit_h1": delta(s1), "sit_h2": delta(s2)}
    save()

    # secondary: own-variance-matched null for each direction
    nulls = {}
    for n, seed, R, base, nn in (("word", SEEDS["null_word"], R_all, base_all, a.n_null),
                                 ("sit", SEEDS["null_sit"], R_all, base_all, a.n_null),
                                 ("pos_even", SEEDS["null_pos"], R_even, base_even, 200)):
        key = "pos" if n == "pos_even" else n
        rs = fast_var_matched(np.random.default_rng(seed), evals, evecs, len(Ac), var[key], nn)
        log.setdefault("null_var_check", {})[n] = float(np.median([np.var(Ac @ r) for r in rs]) / var[key])
        nulls[n] = [delta(r, R, base) for r in rs]
        x, z = obs[n], np.asarray(nulls[n])
        log.setdefault("null", {})[n] = {"n": nn, "mean": float(z.mean()), "q95": float(np.quantile(z, 0.95)),
                                         "p_one": float((np.sum(z >= x) + 1) / (nn + 1))}
        np.save(out / f"s1c_null_{n}_{tag}.npy", z)
        save()

    # primary: item bootstrap of Delta(d_word) - Delta(d_sit)
    rng = np.random.default_rng(SEEDS["boot"])
    diffs = []
    for _ in range(a.n_boot):
        idx = rng.choice(len(items), len(items), replace=True)
        w, s, _ = dirs_from(idx)
        diffs.append(delta(w) - delta(s))
    diffs = np.asarray(diffs)
    np.save(out / f"s1c_boot_{tag}.npy", diffs)
    log["primary"] = {"diff_obs": obs["word"] - obs["sit"], "boot_mean": float(diffs.mean()),
                      "lower95_one_sided": float(np.quantile(diffs, 0.05)),
                      "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
                      "H1c_supported": bool(np.quantile(diffs, 0.05) > 0)}
    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log[k] for k in ("layer", "delta", "null", "primary", "split_half_cos")}, indent=1, default=float))


if __name__ == "__main__":
    main()
