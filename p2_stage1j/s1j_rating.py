"""Stage 1j: a 1-9 rating readout with no option words, plus an unrelated object-weight rating as control.

readout_R9s.csv (build_r9s.py): task S = the 100 B2 items (scenario and frame type) x {mild, severe} added sentence
(200 items); task O = 100 object items. Format "qa" (chosen by the baseline-only format probes, see the
preregistration): "Scenario: ...\nQuestion: On a scale from 1 (...) to 9 (...), ...? Answer with one digit.\nAnswer:".
Forward scale only (the probes showed that no model follows swapped anchors). In all six models " k" tokenises as
[space, digit], so one forward pass of the prefix gives p(digit = 1..9) as the next-token distribution over the nine
digit ids (renormalised). Score per item = E[digit]. Delta = mean over items of steered minus baseline score.
Directions are rebuilt exactly as in Stage 1g (cos check with the saved ones), plus two derived directions:
  val_eval = unit(word_eval - word_poseval); pc1_val = pc1_neg with its component along unit(word_joy + word_poseval)
  removed (the part not shared with positive words).
Nulls: variance-matched random directions; each null direction is scored on S and O in the same pass and saved in
order (index-paired), so S - O contrasts use paired nulls.
Checks (no steering), per model: severity effect = mean E(severe) - E(mild) over the 100 pairs and the fraction of
pairs with a positive difference; object_rho_logweight = Spearman(E on O items, log weight_kg); judgment_rho_B2
(descriptive) = Spearman(mean of mild and severe E per B2 item, pair-demeaned baseline B2).
Positive control: direction from odd S items (hidden state after " 8" minus after " 2"), measured on even S items, 200
nulls.
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
from s1c_word_vs_situation import Readout
from s1d_word_specificity import fast_var_matched

SETS_1D = ("anger", "fear", "joy", "eval")
SETS_1F = ("poseval", "rare")
SEED_BASE = {"pc1_neg": 8001, "word_anger": 8002, "word_eval": 8003, "word_poseval": 8004, "val_eval": 8005,
             "pc1_val": 8006, "pos": 8007}


class Rating:
    """p(digit) at the position after prompt + space token, for every item, in one forward pass per item."""

    def __init__(self, tok, items: list[dict]):
        seqs, digit_ids = [], None
        for it in items:
            full = [tok(it["prompt"] + f" {k}")["input_ids"] for k in range(1, 10)]
            pre = full[0][:-1]
            assert all(f[:-1] == pre for f in full), it["item_id"]
            d = [f[-1] for f in full]
            assert digit_ids is None or d == digit_ids
            digit_ids = d
            seqs.append(pre)
        T = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        self.mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, s in enumerate(seqs):
            self.ids[j, : len(s)] = torch.tensor(s)
            self.mask[j, : len(s)] = 1
        self.last = torch.tensor([len(s) - 1 for s in seqs])
        self.digits = torch.tensor(digit_ids)
        self.ids, self.mask = self.ids.cuda(), self.mask.cuda()

    @torch.no_grad()
    def probs(self, model, steer, v, bs: int = 50) -> np.ndarray:
        steer.v = None if v is None else torch.tensor(v, device="cuda")
        out = []
        for b in range(0, len(self.ids), bs):
            lg = model(input_ids=self.ids[b:b + bs], attention_mask=self.mask[b:b + bs]).logits
            lg = lg[torch.arange(lg.shape[0]), self.last[b:b + bs].cuda()][:, self.digits.cuda()].float()
            out.append(torch.softmax(lg, -1).cpu().numpy())
        steer.v = None
        return np.concatenate(out)

    def score(self, model, steer, v) -> np.ndarray:
        return self.probs(model, steer, v) @ np.arange(1, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1g", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", default="pc1_neg,word_anger,word_eval,word_poseval,val_eval,pc1_val")
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--n_null_minor", type=int, default=500, help="nulls for word_anger, val_eval, pc1_val")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED_BASE, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1j_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

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
    R9 = load_csv(S / "readout_R9s.csv")
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
    u = unit(D["word_joy"] + D["word_poseval"])
    D["pc1_val"] = unit(D["pc1_neg"] - (D["pc1_neg"] @ u) * u)

    j1g = json.load(open(Path(a.res1g) / f"s1g_{tag}.json"))
    saved = dict(zip(j1g["dir_names"], np.load(Path(a.res1g) / f"s1g_dirs_{tag}.npy")))
    log.update({"alpha": alpha, "alpha_1g": j1g["alpha"],
                "cos_with_1g": {n: float(D[n] @ saved[n]) for n in dirs if n in saved},
                "cos_derived": {"val_eval|pc1_neg": float(D["val_eval"] @ D["pc1_neg"]),
                                "pc1_val|pc1_neg": float(D["pc1_val"] @ D["pc1_neg"])},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    save()

    # readouts: S (mild then severe, aligned by B2 item) and O items, qa format
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    Sm = [x for x in R9 if x["sev"] == "mild"]
    Ss = [x for x in R9 if x["sev"] == "severe"]
    Ob = [x for x in R9 if x["task"] == "O"]
    assert [x["scen"] for x in Sm] == [x["scen"] for x in Ss] == [x["item_id"].split("_")[1] for x in R_B2]
    n_s, n_o = len(Sm) + len(Ss), len(Ob)
    Rall = Rating(tok, Sm + Ss + Ob)

    def split(sc):
        return sc[:n_s], sc[n_s:]

    base_raw = Rall.score(model, steer, None)
    base_s, base_o = split(base_raw)
    base_B2 = Readout(tok, R_B2).score(model, steer, None)
    pair = np.array([x["option_a"] if x["target_option"] == "a" else x["option_b"] for x in R_B2])
    dm = lambda v: v - np.array([v[pair == p].mean() for p in pair])
    wkg = np.log10([float(x["weight_kg"]) for x in Ob])
    rho = lambda x, y: float(spearmanr(x, y)[0])
    sev_d = base_s[len(Sm):] - base_s[:len(Sm)]
    log["checks"] = {
        "severity_effect": float(sev_d.mean()), "severity_frac_pos": float((sev_d > 0).mean()),
        "object_rho_logweight": rho(base_o, wkg),
        "judgment_rho_B2_within_pair": rho((base_s[:len(Sm)] + base_s[len(Sm):]) / 2, dm(base_B2)),
        "base_items_raw": base_raw.tolist(), "base_items_B2": base_B2.tolist(),
        "mean_digit_probs_base": Rall.probs(model, steer, None).mean(0).tolist()}
    save()

    def delta(v):
        s, o = split(Rall.score(model, steer, alpha * v))
        return float(np.mean(s - base_s)), float(np.mean(o - base_o))

    # positive control: odd S items -> direction; even S items -> measured
    S_all = Sm + Ss
    odd, even = S_all[0::2], S_all[1::2]
    with torch.no_grad():
        d_pos = unit(last_hidden(model, tok, [x["prompt"] + " 8" for x in odd], L).mean(0)
                     - last_hidden(model, tok, [x["prompt"] + " 2" for x in odd], L).mean(0))
    Reven = Rating(tok, even)
    base_even = Reven.score(model, steer, None)

    def delta_pos(v):
        return float(np.mean(Reven.score(model, steer, alpha * v) - base_even))

    res = log.setdefault("readout", {}).setdefault("R9", {})
    res["base_mean"] = {"S": float(base_s.mean()), "O": float(base_o.mean())}
    dd = {n: delta(D[n]) for n in dirs}
    res["delta_S"] = {n: x[0] for n, x in dd.items()}
    res["delta_O"] = {n: x[1] for n, x in dd.items()}
    res["pos_even"] = {"delta": delta_pos(d_pos)}
    by_sev = {}  # descriptive: Delta_S within mild and within severe items, observed directions only
    for n in dirs:
        s = split(Rall.score(model, steer, alpha * D[n]))[0] - base_s
        by_sev[n] = {"mild": float(s[:len(Sm)].mean()), "severe": float(s[len(Sm):].mean())}
    res["delta_S_by_severity"] = by_sev
    save()

    minor = ("word_anger", "val_eval", "pc1_val")
    for n in dirs + ["pos"]:
        d = d_pos if n == "pos" else D[n]
        nn = 200 if n == "pos" else (a.n_null_minor if n in minor else a.n_null)
        tv = float(np.var(Ac @ d))
        rs = fast_var_matched(np.random.default_rng(SEED_BASE[n]), evals, evecs, len(Ac), tv, nn)
        entry = {"n": nn, "seed": SEED_BASE[n], "var_check": float(np.median([np.var(Ac @ v) for v in rs]) / tv)}
        if n == "pos":
            z = np.asarray([delta_pos(v) for v in rs])
            x = res["pos_even"]["delta"]
            entry.update({"p_one": float((np.sum(z >= x) + 1) / (nn + 1)), "mean": float(z.mean())})
            np.save(out / f"s1j_null_pos_{tag}.npy", z)
        else:
            z = np.asarray([delta(v) for v in rs])  # (nn, 2): S, O
            for j, t in enumerate(("S", "O")):
                x = res[f"delta_{t}"][n]
                entry[t] = {"mean": float(z[:, j].mean()), "sd": float(z[:, j].std(ddof=1)),
                            "p_one": float((np.sum(z[:, j] >= x) + 1) / (nn + 1)),
                            "p_one_lower": float((np.sum(z[:, j] <= x) + 1) / (nn + 1))}
            np.save(out / f"s1j_null_{n}_{tag}.npy", z)
        res.setdefault("null", {})[n] = entry
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("cos_with_1g", "cos_derived", "readout")}, indent=1, default=float))
    print(json.dumps({k: v for k, v in log["checks"].items() if not k.startswith(("base_items", "mean_digit"))}))


if __name__ == "__main__":
    main()
