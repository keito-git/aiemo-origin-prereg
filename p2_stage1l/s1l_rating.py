"""Stage 1l: is the Stage 1k effect specific to negative valence, and does emotional intensity carry it?
Readout and metrics exactly as in Stage 1k (readout_R9k.csv: S = deserved sanction, C = indoors/outdoors, same text;
E = expected digit, slope = OLS slope over k of the change in log p(k)).
Directions (closing-sentence contrasts on the Stage 1c anger items, as in Stages 1d-1k):
  pc1_neg = first SVD axis of {word_anger, word_fear, word_eval}          (as in Stage 1g-1k; cos check with 1g)
  pc1_pos = first SVD axis of {word_joy, word_grat, word_poseval}         (the positive mirror)
  int_neg = unit(word_anger - word_anger_lo)    (furious vs irritated: intensity within negative valence)
  int_pos = unit(word_joy - word_joy_lo)        (thrilled vs pleased: intensity within positive valence)
  val_pc1 = unit(pc1_neg - pc1_pos)             (secondary)
Nulls (variance-matched random directions on the Stage 1k covariance):
  pair: n_pair draws of (r_neg, r_pos); r_neg is matched to var(pc1_neg), r_pos to var(pc1_pos), and
        cos(r_neg, r_pos) is matched to cos(pc1_neg, pc1_pos). So the null of a pc1_neg - pc1_pos contrast keeps the
        dependence between the two directions.
  int_neg, int_pos, val_pc1: independent draws. Positive controls pos_82 / pos_sev as in Stage 1k (200 each).
Manipulation checks (no steering), each closing rated on its own:
  intensity: "how strong is the feeling" 1-9, anger_L vs anger_lo and joy_L vs joy_lo (12 pairs each).
  valence: "how pleasant" 1-9, the negative triple vs the positive triple (36 sentences each).
Saves the direction vectors (R2, Stage 1k) and cos(pc1_neg, pos_sev).
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

# closing set name -> key in the closings json
SETS = {"anger": "anger_L", "fear": "fear_L", "joy": "joy_L", "eval": "eval_L", "poseval": "poseval_L",
        "anger_lo": "anger_lo", "joy_lo": "joy_lo", "grat": "grat_L"}
COV_SETS = ("anger", "fear", "joy", "eval")   # covariance exactly as in Stage 1k
SEED = {"pair": 8301, "int_neg": 8302, "int_pos": 8303, "val_pc1": 8304, "pos_82": 8306, "pos_sev": 8307}
K9 = np.arange(1, 10)
COLS = ("dE_S", "dE_C", "dslope_S", "dslope_C", "dE_S_sev_minus_mild", "dslope_S_sev_minus_mild")
Q_INT = ("On a scale from 1 (a very weak feeling) to 9 (a very strong feeling), how strong is the feeling expressed "
         "in this sentence? Answer with one digit.")
Q_VAL = ("On a scale from 1 (very unpleasant) to 9 (very pleasant), how pleasant is the feeling expressed in this "
         "sentence? Answer with one digit.")


def slope(p_new: np.ndarray, p_old: np.ndarray) -> np.ndarray:
    """Per item OLS slope over k of log p_new(k) - log p_old(k)."""
    d = np.log(np.clip(p_new, 1e-30, None)) - np.log(np.clip(p_old, 1e-30, None))
    kc = K9 - K9.mean()
    return (d - d.mean(1, keepdims=True)) @ kc / (kc @ kc)


def var_matched_coef(c0, evals, n_rows, target_var):
    """Unit coefficient vector (in the eigenbasis) of one variance-matched direction built from the eigenbasis draw c0
    (same rule as fast_var_matched: scale c0 by lam^(beta/2) and choose beta to hit the target variance)."""
    lam = np.clip(evals, 1e-12, None)
    lam_var = evals * (n_rows - 1) / n_rows

    def var_at(beta):
        c = lam ** (beta / 2) * c0
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
    return unit(lam ** (beta / 2) * c0)


def paired_var_matched(rng, evals, evecs, n_rows, tv1, tv2, cos_target, n):
    """n pairs (r1, r2): r1 matched to tv1, r2 matched to tv2, cos(r1, r2) matched to cos_target by bisection on the
    mixing weight t in y2 = t*y1 + sqrt(1-t^2)*y_new (t in [-1, 1]). Works in the eigenbasis (cos is preserved)."""
    d = len(evals)
    out1, out2, got = [], [], []
    for _ in range(n):
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
        out1.append(evecs @ a1)
        out2.append(evecs @ a2)
        got.append(float(a1 @ a2))
    return out1, out2, np.asarray(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1g", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_pair", type=int, default=1000)
    ap.add_argument("--n_null_minor", type=int, default=500)
    ap.add_argument("--n_pos", type=int, default=200)
    ap.add_argument("--probe", action="store_true",
                    help="stop after the manipulation checks and direction cosines (no steering; pre-registration)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1l_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

    S = Path(a.stim)
    items = [json.loads(x) for x in open(S / "raw_generation" / "stage1c_anger.jsonl") if x.strip()]
    cl = (json.load(open(S / "stage1d_closings.json")) | json.load(open(S / "stage1f_closings.json"))
          | json.load(open(S / "stage1l_closings.json")))
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
    R9 = load_csv(S / "readout_R9k.csv")

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

    D = {"pc1_neg": pc1(("anger", "eval", "fear")), "pc1_pos": pc1(("joy", "grat", "poseval")),
         "int_neg": unit(W["anger"] - W["anger_lo"]), "int_pos": unit(W["joy"] - W["joy_lo"])}
    D["val_pc1"] = unit(D["pc1_neg"] - D["pc1_pos"])
    dirs = list(D)

    j1g = json.load(open(Path(a.res1g) / f"s1g_{a.model.split('/')[-1]}_anger_d{a.depth}.json"))
    saved = dict(zip(j1g["dir_names"], np.load(Path(a.res1g) / f"s1g_dirs_{a.model.split('/')[-1]}_anger_d{a.depth}.npy")))
    names = list(D)
    log.update({"alpha": alpha, "alpha_1g": j1g["alpha"],
                "cos_with_1g": {"pc1_neg": float(D["pc1_neg"] @ saved["pc1_neg"]),
                                "word_poseval": float(W["poseval"] @ saved["word_poseval"])},
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(names) for q in names[i + 1:]}
                       | {f"word_{x}|pc1_neg": float(W[x] @ D["pc1_neg"]) for x in SETS}
                       | {f"word_{x}|pc1_pos": float(W[x] @ D["pc1_pos"]) for x in SETS},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    save()

    # manipulation checks: each closing sentence rated on its own (no steering)
    def closing_items(q, key):
        return [{"item_id": f"{key}_{i}", "prompt": f"Sentence: {s}\nQuestion: {q}\nAnswer:"}
                for i, s in enumerate(cl[key])]
    neg_keys, pos_keys = ("anger_L", "fear_L", "eval_L"), ("joy_L", "grat_L", "poseval_L")
    mc = {}
    for hi, lo in (("anger_L", "anger_lo"), ("joy_L", "joy_lo")):
        Rm = Rating(tok, closing_items(Q_INT, hi) + closing_items(Q_INT, lo))
        e = Rm.score(model, steer, None)
        diff = e[:12] - e[12:]
        mc[f"intensity_{hi}_minus_{lo}"] = {"mean": float(diff.mean()), "frac_pos": float((diff > 0).mean()),
                                           "hi": e[:12].tolist(), "lo": e[12:].tolist()}
    Rv = Rating(tok, sum((closing_items(Q_VAL, k) for k in neg_keys + pos_keys), []))
    ev = Rv.score(model, steer, None)
    mc["valence_pos_minus_neg"] = {"mean": float(ev[36:].mean() - ev[:36].mean()),
                                   "by_set": {k: float(ev[12 * i:12 * (i + 1)].mean())
                                              for i, k in enumerate(neg_keys + pos_keys)}}
    log["manipulation"] = mc
    save()
    if a.probe:
        log["t_total_s"] = time.time() - log["t0"]
        save()
        print(json.dumps({k: log.get(k) for k in ("cos_with_1g", "cos", "manipulation")}, indent=1, default=float))
        return

    # readout (as in Stage 1k)
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R9 if x["task"] == t and x["sev"] == s]
    Sm, Ss, Cm, Cs = get("S", "mild"), get("S", "severe"), get("C", "mild"), get("C", "severe")
    assert [x["scen"] for x in Sm] == [x["scen"] for x in Ss] == [x["scen"] for x in Cm] == [x["scen"] for x in Cs]
    n = len(Sm)
    loc = np.array([int(x["loc"]) for x in Sm])
    Rall = Rating(tok, Sm + Ss + Cm + Cs)
    P0 = Rall.probs(model, steer, None)
    E0 = P0 @ K9
    eSm, eSs, eCm, eCs = (E0[i * n:(i + 1) * n] for i in range(4))
    eC = (eCm + eCs) / 2
    log["checks"] = {
        "severity_effect": float((eSs - eSm).mean()), "severity_frac_pos": float(((eSs - eSm) > 0).mean()),
        "C_severity_effect": float((eCs - eCm).mean()), "C_rho_location": float(spearmanr(eC, loc)[0]),
        "S_base_mild": float(eSm.mean()), "S_base_severe": float(eSs.mean()), "C_base_mean": float(eC.mean()),
        "base_E": E0.tolist()}
    save()

    def effects(v) -> np.ndarray:
        P = Rall.probs(model, steer, alpha * v)
        dE, sl = (P - P0) @ K9, slope(P, P0)
        Sx, Cx = slice(0, 2 * n), slice(2 * n, 4 * n)
        return np.array([dE[Sx].mean(), dE[Cx].mean(), sl[Sx].mean(), sl[Cx].mean(),
                         dE[n:2 * n].mean() - dE[:n].mean(), sl[n:2 * n].mean() - sl[:n].mean()])

    odd = np.arange(n) % 2 == 0
    oS = lambda blk: [x["prompt"] for i, x in enumerate(blk) if odd[i]]
    with torch.no_grad():
        d_pos = {"pos_82": unit(last_hidden(model, tok, [p + " 8" for p in oS(Sm) + oS(Ss)], L).mean(0)
                                - last_hidden(model, tok, [p + " 2" for p in oS(Sm) + oS(Ss)], L).mean(0)),
                 "pos_sev": unit(last_hidden(model, tok, oS(Ss), L).mean(0) - last_hidden(model, tok, oS(Sm), L).mean(0))}
    log["cos"] |= {f"{p}|{q}": float(d_pos[p] @ D[q]) for p in d_pos for q in D}
    np.save(out / f"s1l_dirs_{tag}.npy", np.stack([D[k] for k in dirs] + [d_pos[k] for k in d_pos]))
    log["dir_names"] = dirs + list(d_pos)
    even = [x for i, x in enumerate(Sm) if not odd[i]] + [x for i, x in enumerate(Ss) if not odd[i]]
    Reven = Rating(tok, even)
    base_even = Reven.score(model, steer, None)

    def delta_pos(v):
        return float(np.mean(Reven.score(model, steer, alpha * v) - base_even))

    res = log.setdefault("readout", {}).setdefault("R9k", {})
    res["effects"] = {nm: dict(zip(COLS, effects(D[nm]).tolist())) for nm in dirs}
    res["pos_even"] = {k: delta_pos(v) for k, v in d_pos.items()}
    save()

    # positive controls first (gate), then the paired null, then the others
    for nm, v in d_pos.items():
        tv = float(np.var(Ac @ v))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, a.n_pos)
        z = np.asarray([delta_pos(r) for r in rs])
        np.save(out / f"s1l_null_{nm}_{tag}.npy", z)
        res.setdefault("null", {})[nm] = {"n": a.n_pos, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv),
                                          "p_one": float((np.sum(z >= res["pos_even"][nm]) + 1) / (a.n_pos + 1)),
                                          "mean": float(z.mean())}
        save()

    tv1, tv2 = float(np.var(Ac @ D["pc1_neg"])), float(np.var(Ac @ D["pc1_pos"]))
    c12 = float(D["pc1_neg"] @ D["pc1_pos"])
    r1, r2, got = paired_var_matched(np.random.default_rng(SEED["pair"]), evals, evecs, len(Ac), tv1, tv2, c12,
                                     a.n_pair)
    z1 = np.stack([effects(v) for v in r1])
    np.save(out / f"s1l_null_pair_neg_{tag}.npy", z1)
    save()
    z2 = np.stack([effects(v) for v in r2])
    np.save(out / f"s1l_null_pair_pos_{tag}.npy", z2)
    res["null"]["pair"] = {"n": a.n_pair, "cos_target": c12, "cos_got_median": float(np.median(got)),
                           "cos_got_max_abs_err": float(np.max(np.abs(got - c12))),
                           "var_check_neg": float(np.median([np.var(Ac @ v) for v in r1]) / tv1),
                           "var_check_pos": float(np.median([np.var(Ac @ v) for v in r2]) / tv2)}
    save()

    for nm in ("int_neg", "int_pos", "val_pc1"):
        tv = float(np.var(Ac @ D[nm]))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, a.n_null_minor)
        z = np.stack([effects(v) for v in rs])
        np.save(out / f"s1l_null_{nm}_{tag}.npy", z)
        res["null"][nm] = {"n": a.n_null_minor, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv)}
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("cos_with_1g", "manipulation")}, indent=1, default=float))
    print(json.dumps({k: v for k, v in log["checks"].items() if k != "base_E"}))
    print(json.dumps(res["effects"], indent=1))


if __name__ == "__main__":
    main()
