"""Stage 1o: is the effect of the anger intensity direction invariant across readouts that ask for the same sanction
judgment in different formats and words? (design 02_設計/11_段階1o_読み出し不変性_設計_draft_20261103.md)
Readouts (readout_R9o_main.csv: 60 new scenarios L01..L60 x mild/severe; never used in earlier stages):
  judgment  S, HARSH, L (digit 1-9), CAT_S, CAT_S2 (4-choice, four letter rotations), PAIR (2-choice, adjacent ranks,
            both orders); controls  U_bright, U_fam (digit), CAT_U, CAT_R (4-choice).
Every readout is linear in the log choice probabilities (see s1o_gradpos_pilot.py): digit = OLS slope over k of the
change in log p(k) (Stages 1k-1n); 4-choice = slope over semantic rank; PAIR = change in the log-odds of the heavier
option. The effect of a direction = the item mean of that change. PAIR is also kept per adjacent pair.
Directions (same layer, alpha and covariance as Stage 1n; built from the Stage 1c texts, not from the readout texts):
  int_neg = anger_L - anger_lo, int_pos = joy_L - joy_lo, pc1_neg as Stage 1n (cos with Stage 1n checked);
  int_neu = meaB_hi - meaB_lo (gauge near the top vs the middle of its range; stage1o_neu_closings.json), new.
Nulls: pair (r_neg, r_neu) as Stage 1n (cos matched, n_pair draws, seed 8801); int_pos and pc1_neg n_null_minor draws
(descriptive). Every null effect is saved for all items and for the even scenarios (positive controls).
Positive controls (necessary-condition check, design §4.1): for each readout, unit(mean gradient of the readout score
over the odd scenarios), applied at alpha and read out on the even scenarios; compared with the pair_neg nulls on the
even scenarios.
Gradient split (design §6, §10): mean gradient of every readout over all items at delta = 0; projections A(v) =
alpha v.g for the four directions and every null (linear, no forward pass).
Lexical check (descriptive): cos of each direction with the mean hidden state of prompt + rank-4 option minus prompt +
rank-1 option (CAT_S, CAT_S2), against the pair_neg nulls.
No-steering checks: the manipulation checks of Stage 1n with the meaB pair, and the base scores of every readout.
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
from s1d_word_specificity import fast_var_matched
from s1j_rating import Rating
from s1m_rating import COV_SETS, PAIR_TOL, Q_EMO, Q_EXT, Q_INT, Q_VAL, SETS, paired_var_matched_rej

SEED = {"pair": 8801, "int_pos": 8802, "pc1_neg": 8803}
JUDG = ("S", "HARSH", "L", "CAT_S", "CAT_S2", "PAIR")
CTRL = ("U_bright", "U_fam", "CAT_U", "CAT_R")
DIGIT = ("S", "HARSH", "L", "U_bright", "U_fam")
CAT4 = ("CAT_S", "CAT_S2", "CAT_U", "CAT_R")
READ = JUDG + CTRL
PAIRS = ("12", "23", "34")
# effect columns: every readout on all items, PAIR per adjacent pair, every readout on the even scenarios
COLS = READ + tuple(f"PAIR_{p}" for p in PAIRS) + tuple(f"even_{t}" for t in READ)
K9 = np.arange(1, 10)
W9 = (K9 - K9.mean()) / ((K9 - K9.mean()) @ (K9 - K9.mean()))
OPT_HI_LO = {"CAT_S": ("being barred from the place", "no action"),
             "CAT_S2": ("they are not allowed to come back", "nothing further")}   # build_r9o_catprobe*.py
SETS_O = SETS | {"meaB_hi": "meaB_hi", "meaB_lo": "meaB_lo"}


class Readout:
    """Items of one readout, batched in order of length; the choice tokens and per-item weights."""

    def __init__(self, tok, items, choices: list[str], W: np.ndarray):
        seqs, ids = [], None
        for it in items:
            full = [tok(it["prompt"] + f" {k}")["input_ids"] for k in choices]
            pre = full[0][:-1]
            assert all(f[:-1] == pre for f in full), it["item_id"]
            d = [f[-1] for f in full]
            assert ids is None or d == ids
            ids = d
            seqs.append(pre)
        self.order = np.argsort([len(s) for s in seqs], kind="stable")
        self.seqs = [seqs[i] for i in self.order]
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.choice = torch.tensor(ids).cuda()
        self.W = W
        self.even = np.array([int(x["scen"][1:]) % 2 == 0 for x in items])
        self.items = items

    def batch(self, b, e):
        s = self.seqs[b:e]
        T = max(len(x) for x in s)
        ids = torch.full((len(s), T), self.pad, dtype=torch.long)
        mask = torch.zeros((len(s), T), dtype=torch.long)
        for j, x in enumerate(s):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        return ids.cuda(), mask.cuda(), torch.tensor([len(x) - 1 for x in s]).cuda()

    @torch.no_grad()
    def logp(self, model, steer, v, bs: int = 64) -> np.ndarray:
        """log softmax over the choice tokens, items in the original order."""
        steer.v = None if v is None else torch.tensor(v, device="cuda")
        out = []
        for b in range(0, len(self.seqs), bs):
            ids, mask, last = self.batch(b, b + bs)
            lg = model(input_ids=ids, attention_mask=mask).logits
            out.append(torch.log_softmax(lg[torch.arange(len(last)), last][:, self.choice].float(), -1).cpu().numpy())
        steer.v = None
        P = np.empty((len(self.seqs), len(self.choice)))
        P[self.order] = np.concatenate(out)
        return P

    def grads(self, model, steer, d: int, sel: np.ndarray, bs: int = 8) -> np.ndarray:
        """Sum over the selected items of d score_i / d delta (delta added at every position; fp32)."""
        pos = np.flatnonzero(sel[self.order])           # positions in the sorted order
        acc = np.zeros(d)
        for b in range(0, len(pos), bs):
            p = pos[b:b + bs]
            s = [self.seqs[i] for i in p]
            T = max(len(x) for x in s)
            ids = torch.full((len(s), T), self.pad, dtype=torch.long)
            mask = torch.zeros((len(s), T), dtype=torch.long)
            for j, x in enumerate(s):
                ids[j, :len(x)] = torch.tensor(x)
                mask[j, :len(x)] = 1
            last = torch.tensor([len(x) - 1 for x in s]).cuda()
            w = torch.tensor(self.W[self.order[p]], device="cuda", dtype=torch.float32)
            delta = torch.zeros((len(s), 1, d), device="cuda", dtype=torch.float32, requires_grad=True)
            steer.v = delta
            with torch.enable_grad():
                lg = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
                obj = (torch.log_softmax(lg[torch.arange(len(s)), last][:, self.choice].float(), -1) * w).sum()
                obj.backward()
            acc += delta.grad[:, 0].sum(0).detach().cpu().numpy().astype(np.float64)
            steer.v = None
            del lg, obj, delta
        return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1n", required=True, help="Stage 1n result dir (cos check of the directions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_pair", type=int, default=500)
    ap.add_argument("--n_null_minor", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED, "pair_tol": PAIR_TOL, "readouts": READ, "cols": COLS, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1o_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = (json.load(open(S / "stage1d_closings.json")) | json.load(open(S / "stage1f_closings.json"))
          | json.load(open(S / "stage1l_closings.json")) | json.load(open(S / "stage1m_closings.json"))
          | json.load(open(S / "stage1o_neu_closings.json")))
    K = len(cl["neutral_C"])
    assert len(cl["meaB_hi"]) == len(cl["meaB_lo"]) == K
    ks = np.array([i % K for i in range(len(items))])
    texts = {}
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.setdefault(f"{sit}C", []).append(f"{it[sit]} {cl['neutral_C'][ks[i]]}")
            for x, key in SETS_O.items():
                texts.setdefault(f"{sit}{x}", []).append(f"{it[sit]} {cl[key][ks[i]]}")
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    R9 = load_csv(S / "readout_R9o_main.csv")
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    L = int(round(a.depth * cfg.num_hidden_layers))
    log["layer"] = L
    steer = Steer(decoder_layers(model)[L - 1])

    with torch.no_grad():
        H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
        H_ctx = last_hidden(model, tok, C_ctx, L)
    hdim = H_ctx.shape[1]
    alpha = a.alpha_frac * float(np.linalg.norm(H_ctx, axis=1).mean())
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + COV_SETS])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))
    m = {k: v.mean(0) for k, v in H.items()}
    W = {x: unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS_O}

    def pc1(names):
        M = np.stack([W[x] for x in names])
        pc = np.linalg.svd(M, full_matrices=False)[2][0]
        return unit(pc if pc @ M.mean(0) > 0 else -pc)

    D = {"int_neg": unit(W["anger"] - W["anger_lo"]), "int_neu": unit(W["meaB_hi"] - W["meaB_lo"]),
         "int_pos": unit(W["joy"] - W["joy_lo"]), "pc1_neg": pc1(("anger", "eval", "fear"))}
    dirs = list(D)
    np.save(out / f"s1o_dirs_{tag}.npy", np.stack([D[k] for k in dirs]))
    log.update({"alpha": alpha, "dir_names": dirs,
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(dirs) for q in dirs[i + 1:]},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    f1n = Path(a.res1n) / f"s1n_dirs_{tag}.npy"
    if f1n.exists():
        j1n = json.load(open(Path(a.res1n) / f"s1n_{tag}.json"))
        saved = dict(zip(j1n["dir_names"], np.load(f1n)))
        log["check_1n"] = {"alpha_abs_diff": abs(alpha - j1n["alpha"]),
                           "cos_with_1n": {k: float(D[k] @ saved[k]) for k in ("int_neg", "int_pos", "pc1_neg")},
                           "cos_int_neu_1n_1o": float(D["int_neu"] @ saved["int_neu"])}
    save()

    # manipulation checks (no steering), as Stage 1n with the meaB pair as the neutral pair
    def closing_items(q, key):
        return [{"item_id": f"{key}_{i}", "prompt": f"Sentence: {s}\nQuestion: {q}\nAnswer:"}
                for i, s in enumerate(cl[key])]

    def hi_lo(q, hi, lo):
        e = Rating(tok, closing_items(q, hi) + closing_items(q, lo)).score(model, steer, None)
        diff = e[:K] - e[K:]
        return {"mean": float(diff.mean()), "frac_pos": float((diff > 0).mean()), "hi": e[:K].tolist(),
                "lo": e[K:].tolist()}

    mc = {}
    for hi, lo in (("anger_L", "anger_lo"), ("joy_L", "joy_lo")):
        mc[f"intensity_{hi}_minus_{lo}"] = hi_lo(Q_INT, hi, lo)
    for hi, lo in (("anger_L", "anger_lo"), ("joy_L", "joy_lo"), ("meaB_hi", "meaB_lo")):
        mc[f"extremity_{hi}_minus_{lo}"] = hi_lo(Q_EXT, hi, lo)
    emo_keys = ("meaB_hi", "meaB_lo", "neutral_C", "anger_L", "anger_lo", "joy_L", "joy_lo")
    ee = Rating(tok, sum((closing_items(Q_EMO, k) for k in emo_keys), [])).score(model, steer, None)
    emo = {k: ee[K * i:K * (i + 1)] for i, k in enumerate(emo_keys)}
    dn = emo["meaB_hi"] - emo["meaB_lo"]
    mc["emotion"] = {"by_set": {k: float(v.mean()) for k, v in emo.items()},
                     "neu_hi_minus_lo": float(dn.mean()), "neu_hi_minus_lo_frac_pos": float((dn > 0).mean()),
                     "anger_L_minus_lo": float((emo["anger_L"] - emo["anger_lo"]).mean())}
    neg_keys, pos_keys = ("anger_L", "fear_L", "eval_L"), ("joy_L", "grat_L", "poseval_L")
    ev = Rating(tok, sum((closing_items(Q_VAL, k) for k in neg_keys + pos_keys), [])).score(model, steer, None)
    mc["valence_pos_minus_neg"] = {"mean": float(ev[3 * K:].mean() - ev[:3 * K].mean()),
                                   "by_set": {k: float(ev[K * i:K * (i + 1)].mean())
                                              for i, k in enumerate(neg_keys + pos_keys)}}
    log["manipulation"] = mc
    save()

    # readouts
    blk = lambda t, rot="": [x for x in R9 if x["task"] == t and x["rot"] == rot]
    ref = [(x["scen"], x["sev"]) for x in blk("S")]
    RO = {}
    for t in DIGIT:
        it = blk(t)
        assert [(x["scen"], x["sev"]) for x in it] == ref, t
        RO[t] = Readout(tok, it, [str(k) for k in K9], np.tile(W9, (len(it), 1)))
    for t in CAT4:
        it = sum((blk(t, str(r)) for r in range(4)), [])
        assert all([(x["scen"], x["sev"]) for x in blk(t, str(r))] == ref for r in range(4)), t
        rk = np.array([[int(c) for c in x["ranks"]] for x in it], dtype=float)
        RO[t] = Readout(tok, it, list("ABCD"), (rk - 2.5) / 5.0)
    it = sum((blk("PAIR", f"{p}{o}") for p in PAIRS for o in "ab"), [])
    assert len(it) == 720 and all([(x["scen"], x["sev"]) for x in blk("PAIR", f"{p}{o}")] == ref
                                  for p in PAIRS for o in "ab")
    rk = np.array([[int(c) for c in x["ranks"]] for x in it], dtype=float)
    Wp = np.where(rk == rk.max(1, keepdims=True), 1.0, -1.0)
    assert all((Wp[i, 1] == 1.0) == (x["rot"][-1] == "a") for i, x in enumerate(it))   # 'a' order: heavier at B
    RO["PAIR"] = Readout(tok, it, list("AB"), Wp)
    pair_of = np.array([x["rot"][:2] for x in RO["PAIR"].items])

    base = {t: r.logp(model, steer, None) for t, r in RO.items()}
    # base scores per scenario x severity (validity checks are applied in s1o_pooled_tests.py)
    sc = {}
    for t in DIGIT:
        sc[t] = (np.exp(base[t]) @ K9).tolist()
    for t in CAT4:
        rk = (RO[t].W * 5.0 + 2.5)
        E = (np.exp(base[t]) * rk).sum(1).reshape(4, 120)
        sc[t] = E.mean(0).tolist()
        sc[f"{t}_by_rot"] = E.tolist()
    lo_ = (base["PAIR"] * RO["PAIR"].W).sum(1).reshape(6, 120)
    sc["PAIR"] = lo_.mean(0).tolist()
    sc["PAIR_by_rot"] = lo_.tolist()
    log["base"] = {"ref": ref, "scores": sc,
                   "letter_mass_note": "choice probabilities are renormalized among the choice tokens"}
    save()

    def effects(v, per_item: bool = False):
        o, items_ = {}, {}
        for t in READ:
            r = RO[t]
            d = ((r.logp(model, steer, alpha * v) - base[t]) * r.W).sum(1)
            o[t] = d.mean()
            o[f"even_{t}"] = d[r.even].mean()
            if t == "PAIR":
                for p in PAIRS:
                    o[f"PAIR_{p}"] = d[pair_of == p].mean()
            items_[t] = d
        vec = np.array([o[c] for c in COLS])
        return (vec, items_) if per_item else vec

    res = log.setdefault("res", {})
    res["effects"] = {}
    per_item = {}
    for nm in dirs:
        vec, it_ = effects(D[nm], per_item=True)
        res["effects"][nm] = dict(zip(COLS, vec.tolist()))
        per_item[nm] = it_
    np.savez(out / f"s1o_items_{tag}.npz", **{f"{nm}__{t}": per_item[nm][t] for nm in dirs for t in READ})
    save()

    # gradients: odd scenarios (positive controls) and even scenarios (gradient split = their mean)
    t1 = time.time()
    G_odd = {t: RO[t].grads(model, steer, hdim, ~RO[t].even) for t in READ}
    G_even = {t: RO[t].grads(model, steer, hdim, RO[t].even) for t in READ}
    n_it = {t: len(RO[t].W) for t in READ}
    G = {t: (G_odd[t] + G_even[t]) / n_it[t] for t in READ}
    dpos = {t: unit(G_odd[t]) for t in READ}
    np.save(out / f"s1o_grads_{tag}.npy", np.stack([G[t] for t in READ] + [dpos[t] for t in READ]))
    log["grad"] = {"t_s": time.time() - t1, "norm": {t: float(np.linalg.norm(G[t])) for t in READ},
                   "cos": {f"{p}|{q}": float(unit(G[p]) @ unit(G[q])) for i, p in enumerate(READ)
                           for q in READ[i + 1:]},
                   "var_ratio_pos": {t: float(np.var(Ac @ v) / log["var"]["int_neg"]) for t, v in dpos.items()}}
    # gradient split (design §6, §10): remove span of the control gradients; sign-aligned judgment residuals
    Q = np.linalg.qr(np.stack([G[t] for t in CTRL]).T)[0]
    sgn = {t: (-1.0 if t == "L" else 1.0) for t in JUDG}
    r_ = {t: sgn[t] * (G[t] - Q @ (Q.T @ G[t])) for t in JUDG}
    shared = np.mean([r_[t] for t in JUDG], 0)
    digit_sp = np.mean([r_[t] for t in ("S", "HARSH", "L")], 0) - shared
    choice_sp = np.mean([r_[t] for t in ("CAT_S", "CAT_S2", "PAIR")], 0) - shared
    comp = {"shared": shared, "digit_specific": digit_sp, "choice_specific": choice_sp}
    log["grad"]["split"] = {"norm_frac_of_mean_judgment_grad": {k: float(np.linalg.norm(v) / np.mean(
        [np.linalg.norm(G[t]) for t in JUDG])) for k, v in comp.items()},
        "resid_cos": {f"{p}|{q}": float(unit(r_[p]) @ unit(r_[q])) for i, p in enumerate(JUDG) for q in JUDG[i + 1:]}}

    def proj(V):
        V = np.atleast_2d(V)
        return np.stack([alpha * V @ comp[k] for k in comp] + [alpha * V @ G[t] for t in READ], 1)

    PROJ_COLS = tuple(comp) + tuple(f"pred_{t}" for t in READ)
    log["grad"]["proj_cols"] = PROJ_COLS
    log["grad"]["proj_dirs"] = {nm: dict(zip(PROJ_COLS, proj(D[nm])[0].tolist())) for nm in dirs}

    # lexical check (descriptive): option-text direction of CAT_S and CAT_S2
    lex = {}
    with torch.no_grad():
        for t, (hi, lo) in OPT_HI_LO.items():
            pr = [x["prompt"] for x in blk(t, "0")]
            lex[t] = unit(last_hidden(model, tok, [p + " " + hi for p in pr], L).mean(0)
                          - last_hidden(model, tok, [p + " " + lo for p in pr], L).mean(0))
    log["lexical"] = {"cos_dirs": {nm: {t: float(D[nm] @ v) for t, v in lex.items()} for nm in dirs}}

    # positive controls on the even scenarios
    ev_idx = [COLS.index(f"even_{u}") for u in READ]
    pm = {t: effects(dpos[t])[ev_idx] for t in READ}
    log["pos"] = {"obs": {t: float(pm[t][i]) for i, t in enumerate(READ)},
                  "matrix_even": {t: dict(zip(READ, pm[t].tolist())) for t in READ}}
    save()

    # paired null
    tv1, tv2 = float(np.var(Ac @ D["int_neg"])), float(np.var(Ac @ D["int_neu"]))
    c12 = float(D["int_neg"] @ D["int_neu"])
    r1, r2, got, n_rej = paired_var_matched_rej(np.random.default_rng(SEED["pair"]), evals, evecs, len(Ac), tv1, tv2,
                                                c12, a.n_pair)
    res["null"] = {"pair": {"n": a.n_pair, "cos_target": c12, "n_redrawn": n_rej,
                            "cos_got_median": float(np.median(got)),
                            "cos_got_max_abs_err": float(np.max(np.abs(got - c12))),
                            "var_check_neg": float(np.median([np.var(Ac @ v) for v in r1]) / tv1),
                            "var_check_neu": float(np.median([np.var(Ac @ v) for v in r2]) / tv2)}}
    np.save(out / f"s1o_pair_cos_{tag}.npy", got)
    R1, R2 = np.stack(r1), np.stack(r2)
    np.save(out / f"s1o_proj_null_pair_neg_{tag}.npy", proj(R1))
    np.save(out / f"s1o_proj_null_pair_neu_{tag}.npy", proj(R2))
    np.save(out / f"s1o_lex_null_pair_neg_{tag}.npy", np.stack([R1 @ lex[t] for t in lex], 1))
    save()
    for nm, RR in (("pair_neg", r1), ("pair_neu", r2)):
        z = np.full((len(RR), len(COLS)), np.nan)
        for i, v in enumerate(RR):
            z[i] = effects(v)
            if i % 50 == 49 or i == len(RR) - 1:
                np.save(out / f"s1o_null_{nm}_{tag}.npy", z[:i + 1])
        log["res"]["null"][nm] = {"done": len(RR)}
        save()
    for nm in ("int_pos", "pc1_neg"):
        tv = float(np.var(Ac @ D[nm]))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, a.n_null_minor)
        np.save(out / f"s1o_null_{nm}_{tag}.npy", np.stack([effects(v) for v in rs]))
        np.save(out / f"s1o_proj_null_{nm}_{tag}.npy", proj(np.stack(rs)))
        res["null"][nm] = {"n": a.n_null_minor, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv)}
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("check_1n", "cos")}, indent=1))
    print(json.dumps(res["effects"]["int_neg"], indent=1))


if __name__ == "__main__":
    main()
