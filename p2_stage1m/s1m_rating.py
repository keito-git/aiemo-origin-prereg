"""Stage 1m: does the intensity effect of Stage 1l make the judgment harsher, and is it emotional?
Readout readout_R9m.csv: the Stage 1k items (S = deserved sanction, C = indoors/outdoors) plus task L = the same
scenario with a leniency question (higher digit = milder). E = expected digit, slope = OLS slope over k of the change in
log p(k), as in Stage 1k/1l.
  harsher judgment            -> S up, L down
  "more of whatever is asked" -> S up, L up
Directions (closing-sentence contrasts on the Stage 1c anger items, as in Stages 1d-1l):
  int_neg = unit(word_anger - word_anger_lo), int_pos = unit(word_joy - word_joy_lo)   (Stage 1l; cos check with 1l)
  int_neu = unit(word_neu_hi - word_neu_lo)   non-emotional intensity (enormous vs large), stage1m_closings.json
  int_com = unit(int_neg + int_pos), int_dif = unit(int_neg - int_pos)   shared and valence-specific intensity parts
  pc1_neg, pc1_pos as in Stage 1l.
Nulls (variance-matched random directions on the Stage 1k covariance):
  pair: n_pair draws of (r_neg, r_neu); r_neg matched to var(int_neg), r_neu to var(int_neu), cos(r_neg, r_neu) matched
        to cos(int_neg, int_neu). A draw whose cos misses the target by more than PAIR_TOL is redrawn (Stage 1l
        deviation); the number of redraws is recorded.
  int_pos: n_null_minor independent draws; int_com, int_dif, pc1_neg, pc1_pos: n_null_sec each.
  Positive controls pos_82 / pos_sev on S as in Stage 1k/1l.
Manipulation checks (no steering), each closing rated on its own:
  Q_INT  feeling strength, anger_L vs anger_lo and joy_L vs joy_lo (as Stage 1l).
  Q_EXT  how extreme / intense is what the sentence describes, for the anger, joy and neutral pairs.
  Q_EMO  how much emotion the sentence expresses, for neu_hi, neu_lo, neutral_C, anger_L, anger_lo, joy_L, joy_lo.
  Q_VAL  pleasantness, negative vs positive triple (as Stage 1l).
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
from s1l_rating import var_matched_coef

SETS = {"anger": "anger_L", "fear": "fear_L", "joy": "joy_L", "eval": "eval_L", "poseval": "poseval_L",
        "anger_lo": "anger_lo", "joy_lo": "joy_lo", "grat": "grat_L", "neu_hi": "neu_hi", "neu_lo": "neu_lo"}
COV_SETS = ("anger", "fear", "joy", "eval")   # covariance exactly as in Stage 1k/1l
SEED = {"pair": 8401, "int_pos": 8402, "int_com": 8403, "int_dif": 8404, "pc1_neg": 8405, "pc1_pos": 8406,
        "pos_82": 8407, "pos_sev": 8408}
PAIR_TOL = 0.01
K9 = np.arange(1, 10)
COLS = ("dE_S", "dE_C", "dE_L", "dslope_S", "dslope_C", "dslope_L")
Q_INT = ("On a scale from 1 (a very weak feeling) to 9 (a very strong feeling), how strong is the feeling expressed "
         "in this sentence? Answer with one digit.")
Q_EXT = ("On a scale from 1 (not extreme at all) to 9 (extremely extreme), how extreme or intense is what this "
         "sentence describes? Answer with one digit.")
Q_EMO = ("On a scale from 1 (no emotion at all) to 9 (a very strong emotion), how much emotion does this sentence "
         "express? Answer with one digit.")
Q_VAL = ("On a scale from 1 (very unpleasant) to 9 (very pleasant), how pleasant is the feeling expressed in this "
         "sentence? Answer with one digit.")


def slope(p_new: np.ndarray, p_old: np.ndarray) -> np.ndarray:
    """Per item OLS slope over k of log p_new(k) - log p_old(k)."""
    d = np.log(np.clip(p_new, 1e-30, None)) - np.log(np.clip(p_old, 1e-30, None))
    kc = K9 - K9.mean()
    return (d - d.mean(1, keepdims=True)) @ kc / (kc @ kc)


def paired_var_matched_rej(rng, evals, evecs, n_rows, tv1, tv2, cos_target, n, tol=PAIR_TOL):
    """As s1l_rating.paired_var_matched, but a pair whose cos misses the target by more than tol is redrawn.
    Returns (r1 list, r2 list, cos of the kept pairs, number of redraws)."""
    d = len(evals)
    out1, out2, got, n_rej = [], [], [], 0
    while len(out1) < n:
        y1, yn = rng.standard_normal(d), rng.standard_normal(d)
        c1, cn = evecs.T @ y1, evecs.T @ yn
        a1 = var_matched_coef(c1, evals, n_rows, tv1)
        mk = lambda t: var_matched_coef(t * c1 + np.sqrt(max(0.0, 1 - t * t)) * cn, evals, n_rows, tv2)
        lo, hi = -1.0, 1.0
        for _ in range(30):
            t = (lo + hi) / 2
            if float(mk(t) @ a1) < cos_target:
                lo = t
            else:
                hi = t
        a2 = mk((lo + hi) / 2)
        c = float(a1 @ a2)
        if abs(c - cos_target) > tol:
            n_rej += 1
            continue
        out1.append(evecs @ a1)
        out2.append(evecs @ a2)
        got.append(c)
    return out1, out2, np.asarray(got), n_rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1l", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_pair", type=int, default=1000)
    ap.add_argument("--n_null_minor", type=int, default=500)
    ap.add_argument("--n_null_sec", type=int, default=300)
    ap.add_argument("--n_pos", type=int, default=200)
    ap.add_argument("--probe", action="store_true",
                    help="stop after the manipulation checks, cosines and baseline readout checks (no steering)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED, "pair_tol": PAIR_TOL, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1m_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = (json.load(open(S / "stage1d_closings.json")) | json.load(open(S / "stage1f_closings.json"))
          | json.load(open(S / "stage1l_closings.json")) | json.load(open(S / "stage1m_closings.json")))
    K = len(cl["neutral_C"])
    ks = np.array([i % K for i in range(len(items))])
    texts = {}
    for i, it in enumerate(items):
        for sit in ("S", "N"):
            texts.setdefault(f"{sit}C", []).append(f"{it[sit]} {cl['neutral_C'][ks[i]]}")
            for x, key in SETS.items():
                texts.setdefault(f"{sit}{x}", []).append(f"{it[sit]} {cl[key][ks[i]]}")
    ctx = load_csv(S / "contexts.csv")
    C_ctx = [r["text"] for r in ctx if r["split"] == "train"][:128] + [r["text"] for r in ctx if r["split"] == "dev"]
    R9 = load_csv(S / "readout_R9m.csv")

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
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + COV_SETS])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    m = {k: v.mean(0) for k, v in H.items()}
    W = {x: unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS}

    def pc1(names):
        M = np.stack([W[x] for x in names])
        pc = np.linalg.svd(M, full_matrices=False)[2][0]
        return unit(pc if pc @ M.mean(0) > 0 else -pc)

    D = {"int_neg": unit(W["anger"] - W["anger_lo"]), "int_neu": unit(W["neu_hi"] - W["neu_lo"]),
         "int_pos": unit(W["joy"] - W["joy_lo"])}
    D["int_com"] = unit(D["int_neg"] + D["int_pos"])
    D["int_dif"] = unit(D["int_neg"] - D["int_pos"])
    D["pc1_neg"] = pc1(("anger", "eval", "fear"))
    D["pc1_pos"] = pc1(("joy", "grat", "poseval"))
    dirs = list(D)

    j1l = json.load(open(Path(a.res1l) / f"s1l_{tag}.json"))
    saved = dict(zip(j1l["dir_names"], np.load(Path(a.res1l) / f"s1l_dirs_{tag}.npy")))
    log.update({"alpha": alpha, "alpha_1l": j1l["alpha"],
                "cos_with_1l": {k: float(D[k] @ saved[k]) for k in ("int_neg", "int_pos", "pc1_neg", "pc1_pos")},
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(dirs) for q in dirs[i + 1:]}
                       | {f"word_neu_hi|{q}": float(W["neu_hi"] @ D[q]) for q in dirs},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    save()

    # manipulation checks: each closing sentence rated on its own (no steering)
    def closing_items(q, key):
        return [{"item_id": f"{key}_{i}", "prompt": f"Sentence: {s}\nQuestion: {q}\nAnswer:"}
                for i, s in enumerate(cl[key])]

    def hi_lo(q, hi, lo):
        e = Rating(tok, closing_items(q, hi) + closing_items(q, lo)).score(model, steer, None)
        diff = e[:12] - e[12:]
        return {"mean": float(diff.mean()), "frac_pos": float((diff > 0).mean()), "hi": e[:12].tolist(),
                "lo": e[12:].tolist()}

    mc = {}
    for hi, lo in (("anger_L", "anger_lo"), ("joy_L", "joy_lo")):
        mc[f"intensity_{hi}_minus_{lo}"] = hi_lo(Q_INT, hi, lo)
    for hi, lo in (("anger_L", "anger_lo"), ("joy_L", "joy_lo"), ("neu_hi", "neu_lo")):
        mc[f"extremity_{hi}_minus_{lo}"] = hi_lo(Q_EXT, hi, lo)
    emo_keys = ("neu_hi", "neu_lo", "neutral_C", "anger_L", "anger_lo", "joy_L", "joy_lo")
    ee = Rating(tok, sum((closing_items(Q_EMO, k) for k in emo_keys), [])).score(model, steer, None)
    emo = {k: ee[12 * i:12 * (i + 1)] for i, k in enumerate(emo_keys)}
    dn = emo["neu_hi"] - emo["neu_lo"]
    mc["emotion"] = {"by_set": {k: float(v.mean()) for k, v in emo.items()},
                     "neu_hi_minus_lo": float(dn.mean()), "neu_hi_minus_lo_frac_pos": float((dn > 0).mean()),
                     "anger_L_minus_lo": float((emo["anger_L"] - emo["anger_lo"]).mean())}
    neg_keys, pos_keys = ("anger_L", "fear_L", "eval_L"), ("joy_L", "grat_L", "poseval_L")
    ev = Rating(tok, sum((closing_items(Q_VAL, k) for k in neg_keys + pos_keys), [])).score(model, steer, None)
    mc["valence_pos_minus_neg"] = {"mean": float(ev[36:].mean() - ev[:36].mean()),
                                   "by_set": {k: float(ev[12 * i:12 * (i + 1)].mean())
                                              for i, k in enumerate(neg_keys + pos_keys)}}
    log["manipulation"] = mc
    save()

    # readout: S, C, L (no steering checks)
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R9 if x["task"] == t and x["sev"] == s]
    blocks = [get(t, s) for t in ("S", "C", "L") for s in ("mild", "severe")]
    assert all([x["scen"] for x in b] == [x["scen"] for x in blocks[0]] for b in blocks)
    Sm, Ss = blocks[0], blocks[1]
    n = len(Sm)
    loc = np.array([int(x["loc"]) for x in Sm])
    Rall = Rating(tok, sum(blocks, []))
    P0 = Rall.probs(model, steer, None)
    E0 = P0 @ K9
    eSm, eSs, eCm, eCs, eLm, eLs = (E0[i * n:(i + 1) * n] for i in range(6))
    eC = (eCm + eCs) / 2
    log["checks"] = {
        "severity_effect": float((eSs - eSm).mean()), "severity_frac_pos": float(((eSs - eSm) > 0).mean()),
        "C_severity_effect": float((eCs - eCm).mean()), "C_rho_location": float(spearmanr(eC, loc)[0]),
        "L_severity_effect": float((eLs - eLm).mean()), "L_severity_frac_neg": float(((eLs - eLm) < 0).mean()),
        "rho_S_L": float(spearmanr(np.r_[eSm, eSs], np.r_[eLm, eLs])[0]),
        "S_base_mild": float(eSm.mean()), "S_base_severe": float(eSs.mean()), "C_base_mean": float(eC.mean()),
        "L_base_mild": float(eLm.mean()), "L_base_severe": float(eLs.mean()), "base_E": E0.tolist()}
    save()
    if a.probe:
        log["t_total_s"] = time.time() - log["t0"]
        save()
        print(json.dumps({k: log.get(k) for k in ("cos_with_1l", "cos")}, indent=1, default=float))
        print(json.dumps({k: v for k, v in mc.items()}, default=float))
        print(json.dumps({k: v for k, v in log["checks"].items() if k != "base_E"}))
        return

    def effects(v) -> np.ndarray:
        P = Rall.probs(model, steer, alpha * v)
        dE, sl = (P - P0) @ K9, slope(P, P0)
        Sx, Cx, Lx = slice(0, 2 * n), slice(2 * n, 4 * n), slice(4 * n, 6 * n)
        return np.array([dE[Sx].mean(), dE[Cx].mean(), dE[Lx].mean(), sl[Sx].mean(), sl[Cx].mean(), sl[Lx].mean()])

    odd = np.arange(n) % 2 == 0
    oS = lambda blk: [x["prompt"] for i, x in enumerate(blk) if odd[i]]
    with torch.no_grad():
        d_pos = {"pos_82": unit(last_hidden(model, tok, [p + " 8" for p in oS(Sm) + oS(Ss)], L).mean(0)
                                - last_hidden(model, tok, [p + " 2" for p in oS(Sm) + oS(Ss)], L).mean(0)),
                 "pos_sev": unit(last_hidden(model, tok, oS(Ss), L).mean(0) - last_hidden(model, tok, oS(Sm), L).mean(0))}
    log["cos"] |= {f"{p}|{q}": float(d_pos[p] @ D[q]) for p in d_pos for q in D}
    np.save(out / f"s1m_dirs_{tag}.npy", np.stack([D[k] for k in dirs] + [d_pos[k] for k in d_pos]))
    log["dir_names"] = dirs + list(d_pos)
    even = [x for i, x in enumerate(Sm) if not odd[i]] + [x for i, x in enumerate(Ss) if not odd[i]]
    Reven = Rating(tok, even)
    base_even = Reven.score(model, steer, None)

    def delta_pos(v):
        return float(np.mean(Reven.score(model, steer, alpha * v) - base_even))

    res = log.setdefault("readout", {}).setdefault("R9m", {})
    res["effects"] = {nm: dict(zip(COLS, effects(D[nm]).tolist())) for nm in dirs}
    res["pos_even"] = {k: delta_pos(v) for k, v in d_pos.items()}
    save()

    # positive controls first (gate), then the paired null, then the others
    for nm, v in d_pos.items():
        tv = float(np.var(Ac @ v))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, a.n_pos)
        z = np.asarray([delta_pos(r) for r in rs])
        np.save(out / f"s1m_null_{nm}_{tag}.npy", z)
        res.setdefault("null", {})[nm] = {"n": a.n_pos, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv),
                                          "p_one": float((np.sum(z >= res["pos_even"][nm]) + 1) / (a.n_pos + 1)),
                                          "mean": float(z.mean())}
        save()

    tv1, tv2 = float(np.var(Ac @ D["int_neg"])), float(np.var(Ac @ D["int_neu"]))
    c12 = float(D["int_neg"] @ D["int_neu"])
    r1, r2, got, n_rej = paired_var_matched_rej(np.random.default_rng(SEED["pair"]), evals, evecs, len(Ac), tv1, tv2,
                                                c12, a.n_pair)
    res["null"]["pair"] = {"n": a.n_pair, "cos_target": c12, "n_redrawn": n_rej,
                           "cos_got_median": float(np.median(got)),
                           "cos_got_max_abs_err": float(np.max(np.abs(got - c12))),
                           "var_check_neg": float(np.median([np.var(Ac @ v) for v in r1]) / tv1),
                           "var_check_neu": float(np.median([np.var(Ac @ v) for v in r2]) / tv2)}
    np.save(out / f"s1m_pair_cos_{tag}.npy", got)
    save()
    z1 = np.stack([effects(v) for v in r1])
    np.save(out / f"s1m_null_pair_neg_{tag}.npy", z1)
    save()
    z2 = np.stack([effects(v) for v in r2])
    np.save(out / f"s1m_null_pair_neu_{tag}.npy", z2)
    save()

    for nm, nn in (("int_pos", a.n_null_minor), ("int_com", a.n_null_sec), ("int_dif", a.n_null_sec),
                   ("pc1_neg", a.n_null_sec), ("pc1_pos", a.n_null_sec)):
        tv = float(np.var(Ac @ D[nm]))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, nn)
        z = np.stack([effects(v) for v in rs])
        np.save(out / f"s1m_null_{nm}_{tag}.npy", z)
        res["null"][nm] = {"n": nn, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv)}
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("cos_with_1l", "manipulation")}, indent=1, default=float))
    print(json.dumps({k: v for k, v in log["checks"].items() if k != "base_E"}))
    print(json.dumps(res["effects"], indent=1))


if __name__ == "__main__":
    main()
