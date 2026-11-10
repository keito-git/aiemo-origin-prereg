"""Stage 1q: does alignment with the judgment gradient predict, in advance, the judgment effect of new directions?
(design 02_設計/13_段階1q_整列の前向き検証_設計_draft_20261110.md)
Directions: unit(W_hi - W_lo) for twelve new magnitude contrasts (stimuli/build_stage1q_closings.py; pleasant brtK swtL
sftM frgN qutP clrQ, unpleasant nsyR stkS grsT btrU cldV drtW), built exactly as in Stage 1p (layer, alpha, covariance).
Items are split by scenario: A = L01-L30, B = L31-L60.
  1. Gradient of every readout on the A items only (as Stage 1o/1p, delta = 0), its split (shared, digit_specific),
     and the projections of every direction and every null are computed and saved (s1q_pred_{tag}.json and
     s1q_proj_null_*.npy, with their SHA256 in the log) BEFORE any steered forward pass.
  2. Effects of every direction on the B items (the tested effects) and on all items (descriptive).
  3. Nulls: n_null variance-matched random directions per contrast (seed 9100 + index), effects on the B items.
Ratings (no steering, closing sentences alone): ext, emo, int, val, aro, hi - lo, as in Stage 1p.
"""
from __future__ import annotations

import argparse
import hashlib
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
from s1m_rating import COV_SETS, Q_EMO, Q_EXT, Q_INT, Q_VAL, SETS
from s1o_rating import CAT4, COLS, CTRL, DIGIT, JUDG, K9, PAIRS, READ, W9, Readout
from s1p_probe import Q_ARO
from s1q_probe import PAIRS as CONTRAST_KEYS, PLEASANT, UNPLEASANT

CONTRASTS = tuple(c for c, _, _ in CONTRAST_KEYS)
SEED0 = 9100
SCEN_A = tuple(f"L{i:02d}" for i in range(1, 31))
SCEN_B = tuple(f"L{i:02d}" for i in range(31, 61))
assert set(PLEASANT) | set(UNPLEASANT) == set(CONTRASTS)


def build_readouts(tok, R9, scens):
    """As s1p_rating.build_readouts, restricted to the items of the given scenarios."""
    R9 = [x for x in R9 if x["scen"] in scens]
    blk = lambda t, rot="": [x for x in R9 if x["task"] == t and x["rot"] == rot]
    ref = [(x["scen"], x["sev"]) for x in blk("S")]
    assert len(ref) == 2 * len(scens)
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
    assert len(it) == 12 * len(scens)
    rk = np.array([[int(c) for c in x["ranks"]] for x in it], dtype=float)
    Wp = np.where(rk == rk.max(1, keepdims=True), 1.0, -1.0)
    assert all((Wp[i, 1] == 1.0) == (x["rot"][-1] == "a") for i, x in enumerate(it))
    RO["PAIR"] = Readout(tok, it, list("AB"), Wp)
    return RO, np.array([x["rot"][:2] for x in RO["PAIR"].items])


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_d{a.depth}"
    log = {"args": vars(a), "seed0": SEED0, "contrasts": CONTRASTS, "readouts": READ, "cols": COLS, "t0": time.time(),
           "scen_A": SCEN_A, "scen_B": SCEN_B,
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1q_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = {}
    for f in ("stage1d", "stage1f", "stage1l", "stage1m", "stage1o_neu", "stage1p", "stage1q"):
        cl |= json.load(open(S / f"{f}_closings.json"))
    K = len(cl["neutral_C"])
    keys = sorted({k for _, h, l in CONTRAST_KEYS for k in (h, l)} | set(SETS.values()))
    assert all(len(cl[k]) == K for k in keys)
    ks = np.array([i % K for i in range(len(items))])
    texts = {}
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.setdefault(f"{sit}|neutral_C", []).append(f"{it[sit]} {cl['neutral_C'][ks[i]]}")
            for k in keys:
                texts.setdefault(f"{sit}|{k}", []).append(f"{it[sit]} {cl[k][ks[i]]}")
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
    # covariance exactly as Stages 1n/1o/1p: contexts + neutral_C + the COV_SETS closings
    cov_keys = ["neutral_C"] + [SETS[x] for x in COV_SETS]
    Ac = np.concatenate([H_ctx] + [H[f"{s}|{k}"] for k in cov_keys for s in ("S", "N")])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))
    m = {k: v.mean(0) for k, v in H.items()}
    W = {k: unit((m[f"S|{k}"] + m[f"N|{k}"]) / 2 - (m["S|neutral_C"] + m["N|neutral_C"]) / 2) for k in keys}
    D = {c: unit(W[h] - W[l]) for c, h, l in CONTRAST_KEYS}
    np.save(out / f"s1q_dirs_{tag}.npy", np.stack([D[c] for c in CONTRASTS]))
    log.update({"alpha": alpha,
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(CONTRASTS) for q in CONTRASTS[i + 1:]},
                "var": {c: float(np.var(Ac @ D[c])) for c in CONTRASTS}})
    save()

    RO_A, _ = build_readouts(tok, R9, SCEN_A)
    RO_B, pair_B = build_readouts(tok, R9, SCEN_B)
    RO_all, pair_all = build_readouts(tok, R9, SCEN_A + SCEN_B)

    # 1. prediction, fixed before any steered pass: gradient on the A items, projections of directions and nulls
    t1 = time.time()
    G = {t: RO_A[t].grads(model, steer, hdim, np.ones(len(RO_A[t].W), bool)) / len(RO_A[t].W) for t in READ}
    Q = np.linalg.qr(np.stack([G[t] for t in CTRL]).T)[0]
    sgn = {t: (-1.0 if t == "L" else 1.0) for t in JUDG}
    r_ = {t: sgn[t] * (G[t] - Q @ (Q.T @ G[t])) for t in JUDG}
    shared = np.mean([r_[t] for t in JUDG], 0)
    comp = {"shared": shared,
            "digit_specific": np.mean([r_[t] for t in ("S", "HARSH", "L")], 0) - shared}
    np.save(out / f"s1q_grads_A_{tag}.npy", np.stack([G[t] for t in READ] + [comp[k] for k in comp]))

    def proj(V):
        V = np.atleast_2d(V)
        return np.stack([alpha * V @ comp[k] for k in comp] + [alpha * V @ G[t] for t in READ], 1)

    PROJ_COLS = tuple(comp) + tuple(f"pred_{t}" for t in READ)
    nulls = {}
    for i, c in enumerate(CONTRASTS):
        tv = float(np.var(Ac @ D[c]))
        nulls[c] = (tv, fast_var_matched(np.random.default_rng(SEED0 + i), evals, evecs, len(Ac), tv, a.n_null))
        np.save(out / f"s1q_proj_null_{c}_{tag}.npy", proj(np.stack(nulls[c][1])))
    pred = {"proj_cols": PROJ_COLS, "proj_dirs": {c: dict(zip(PROJ_COLS, proj(D[c])[0].tolist())) for c in CONTRASTS}}
    pf = out / f"s1q_pred_{tag}.json"
    pf.write_text(json.dumps(pred, indent=1))
    log["grad"] = {"t_s": time.time() - t1, "norm": {t: float(np.linalg.norm(G[t])) for t in READ},
                   "sha256": {p.name: sha(p) for p in [pf] + [out / f"s1q_proj_null_{c}_{tag}.npy" for c in CONTRASTS]},
                   "fixed_at": time.time()}
    save()

    # ratings of the closing sentences alone (no steering)
    def closing_items(q, key):
        return [{"item_id": f"{key}_{i}", "prompt": f"Sentence: {s}\nQuestion: {q}\nAnswer:"}
                for i, s in enumerate(cl[key])]

    rat = {}
    for c, h, l in CONTRAST_KEYS:
        r = {}
        for qn, q in (("ext", Q_EXT), ("emo", Q_EMO), ("int", Q_INT), ("val", Q_VAL), ("aro", Q_ARO)):
            e = Rating(tok, closing_items(q, h) + closing_items(q, l)).score(model, steer, None)
            d = e[:K] - e[K:]
            r[qn] = {"mean": float(d.mean()), "frac_pos": float((d > 0).mean()), "hi_mean": float(e[:K].mean()),
                     "lo_mean": float(e[K:].mean())}
        rat[c] = r
    log["ratings"] = rat
    save()

    # 2. steered effects
    base = {"B": {t: r.logp(model, steer, None) for t, r in RO_B.items()},
            "all": {t: r.logp(model, steer, None) for t, r in RO_all.items()}}
    log["base"] = {"digit_mean_all": {t: float((np.exp(base["all"][t]) @ K9).mean()) for t in DIGIT}}

    def effects(v, RO, b, pair_of, per_item: bool = False):
        o, items_ = {}, {}
        for t in READ:
            r = RO[t]
            d = ((r.logp(model, steer, alpha * v) - b[t]) * r.W).sum(1)
            o[t] = d.mean()
            o[f"even_{t}"] = d[r.even].mean()
            if t == "PAIR":
                for p in PAIRS:
                    o[f"PAIR_{p}"] = d[pair_of == p].mean()
            items_[t] = d
        vec = np.array([o[c] for c in COLS])
        return (vec, items_) if per_item else vec

    res = log.setdefault("res", {"effects_B": {}, "effects_all": {}, "null": {}})
    per_item = {}
    for c in CONTRASTS:
        res["effects_B"][c] = dict(zip(COLS, effects(D[c], RO_B, base["B"], pair_B).tolist()))
        vec, it_ = effects(D[c], RO_all, base["all"], pair_all, per_item=True)
        res["effects_all"][c] = dict(zip(COLS, vec.tolist()))
        per_item[c] = it_
    np.savez(out / f"s1q_items_{tag}.npz", **{f"{c}__{t}": per_item[c][t] for c in CONTRASTS for t in READ})
    save()

    # 3. nulls on the B items
    for c in CONTRASTS:
        tv, rs = nulls[c]
        z = np.full((len(rs), len(COLS)), np.nan)
        for j, v in enumerate(rs):
            z[j] = effects(v, RO_B, base["B"], pair_B)
            if j % 50 == 49 or j == len(rs) - 1:
                np.save(out / f"s1q_null_{c}_{tag}.npy", z[:j + 1])
        res["null"][c] = {"n": len(rs), "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv)}
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({c: {k: round(v["mean"], 2) for k, v in rat[c].items()} for c in CONTRASTS}, indent=1))


if __name__ == "__main__":
    main()
