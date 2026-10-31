"""Stage 1n: does the intensity effect move the judgment, or every question that asks for a degree (R3 C')?
Readout readout_R9n.csv: the 60 Stage 1k scenarios x mild/severe with seven questions (S, C, L as in Stage 1m; HARSH;
U_bright, U_fam, U_size = degree questions about the place that do not follow the case severity, checked in the
Stage 1n probes). E = expected digit, slope = OLS slope over k of the change in log p(k), as in Stages 1k-1m.
  judgment change              -> S up, U unchanged
  degree projection (C', strong) -> S and every U up
  degree projection (C', weak)   -> the U close in meaning to the direction up (U_size for int_neu: large/enormous)
Directions as in Stage 1m: int_neg, int_neu, int_pos, pc1_neg (cos check with the Stage 1m directions).
Nulls (variance-matched random directions on the Stage 1k covariance, as in Stage 1m):
  pair: n_pair draws of (r_neg, r_neu), cos matched to cos(int_neg, int_neu), redrawn if off by more than PAIR_TOL.
  int_pos, pc1_neg: n_null_minor independent draws each.
  Positive controls (n_pos each): pos_82 and pos_sev on S as in Stage 1m; pos_U82 on U_bright = mean hidden state of
  U_bright prompts + " 8" minus + " 2" on the odd scenarios, read out on the even scenarios.
Manipulation checks (no steering) as in Stage 1m.
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
from s1m_rating import COV_SETS, PAIR_TOL, Q_EMO, Q_EXT, Q_INT, Q_VAL, SETS, paired_var_matched_rej, slope

TASKS = ("S", "C", "L", "HARSH", "U_bright", "U_fam", "U_size")
SEED = {"pair": 8501, "int_pos": 8502, "pc1_neg": 8503, "pos_82": 8504, "pos_sev": 8505, "pos_U82": 8506}
K9 = np.arange(1, 10)
COLS = tuple(f"dE_{t}" for t in TASKS) + tuple(f"dslope_{t}" for t in TASKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_pair", type=int, default=600)
    ap.add_argument("--n_null_minor", type=int, default=300)
    ap.add_argument("--n_pos", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    log = {"args": vars(a), "seeds": SEED, "pair_tol": PAIR_TOL, "tasks": TASKS, "cols": COLS, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}

    def save():
        (out / f"s1n_{tag}.json").write_text(json.dumps(log, indent=1, default=float))

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
    R9 = load_csv(S / "readout_R9n.csv")

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
         "int_pos": unit(W["joy"] - W["joy_lo"]), "pc1_neg": pc1(("anger", "eval", "fear"))}
    dirs = list(D)

    j1m = json.load(open(Path(a.res1m) / f"s1m_{tag}.json"))
    saved = dict(zip(j1m["dir_names"], np.load(Path(a.res1m) / f"s1m_dirs_{tag}.npy")))
    log.update({"alpha": alpha, "alpha_1m": j1m["alpha"],
                "cos_with_1m": {k: float(D[k] @ saved[k]) for k in dirs},
                "cos": {f"{p}|{q}": float(D[p] @ D[q]) for i, p in enumerate(dirs) for q in dirs[i + 1:]},
                "var": {n: float(np.var(Ac @ D[n])) for n in dirs}})
    save()

    # manipulation checks: each closing sentence rated on its own (no steering), as in Stage 1m
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
    log["manipulation_max_abs_diff_1m"] = float(max(
        abs(mc[k]["mean"] - j1m["manipulation"][k]["mean"]) for k in mc if "mean" in mc[k]))
    save()

    # readout: seven questions (no steering checks)
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R9 if x["task"] == t and x["sev"] == s]
    order = [(t, s) for t in TASKS for s in ("mild", "severe")]
    blocks = {k: get(*k) for k in order}
    ref = [x["scen"] for x in blocks[("S", "mild")]]
    assert all([x["scen"] for x in b] == ref for b in blocks.values())
    n = len(ref)
    loc = np.array([int(x["loc"]) for x in blocks[("S", "mild")]])
    Rall = Rating(tok, sum((blocks[k] for k in order), []))
    P0 = Rall.probs(model, steer, None)
    E0 = P0 @ K9
    e = {k: E0[i * n:(i + 1) * n] for i, k in enumerate(order)}
    allS = np.r_[e[("S", "mild")], e[("S", "severe")]]
    tk = {}
    for t in TASKS:
        mi, se = e[(t, "mild")], e[(t, "severe")]
        d = se - mi
        tk[t] = {"severity_effect": float(d.mean()), "frac_pos": float((d > 0).mean()),
                 "frac_neg": float((d < 0).mean()), "rho_mild_severe": float(spearmanr(mi, se)[0]),
                 "rho_with_S": float(spearmanr(np.r_[mi, se], allS)[0]), "mean_mild": float(mi.mean()),
                 "mean_severe": float(se.mean())}
    tk["C"]["rho_location"] = float(spearmanr((e[("C", "mild")] + e[("C", "severe")]) / 2, loc)[0])
    b1m = np.asarray(j1m["checks"]["base_E"])
    # items 350-359 sit in a different batch shape than in Stage 1m (bf16 path; Stage 1n probe), so also report the rest
    log["checks"] = {"tasks": tk, "repro_1m_SCL_max_abs_diff": float(np.max(np.abs(E0[:6 * n] - b1m))),
                     "repro_1m_SCL_first350_max_abs_diff": float(np.max(np.abs(E0[:350] - b1m[:350]))),
                     "base_E": E0.tolist()}
    save()

    task_idx = {t: np.r_[TASKS.index(t) * 2 * n:(TASKS.index(t) + 1) * 2 * n] for t in TASKS}

    def effects(v) -> np.ndarray:
        P = Rall.probs(model, steer, alpha * v)
        dE, sl = (P - P0) @ K9, slope(P, P0)
        return np.array([dE[task_idx[t]].mean() for t in TASKS] + [sl[task_idx[t]].mean() for t in TASKS])

    odd = np.arange(n) % 2 == 0
    oP = lambda t: [x["prompt"] for s in ("mild", "severe") for i, x in enumerate(blocks[(t, s)]) if odd[i]]
    eI = lambda t: [x for s in ("mild", "severe") for i, x in enumerate(blocks[(t, s)]) if not odd[i]]
    with torch.no_grad():
        d_pos = {"pos_82": unit(last_hidden(model, tok, [p + " 8" for p in oP("S")], L).mean(0)
                                - last_hidden(model, tok, [p + " 2" for p in oP("S")], L).mean(0)),
                 "pos_sev": unit(last_hidden(model, tok, [x["prompt"] for i, x in enumerate(blocks[("S", "severe")])
                                                          if odd[i]], L).mean(0)
                                 - last_hidden(model, tok, [x["prompt"] for i, x in enumerate(blocks[("S", "mild")])
                                                            if odd[i]], L).mean(0)),
                 "pos_U82": unit(last_hidden(model, tok, [p + " 8" for p in oP("U_bright")], L).mean(0)
                                 - last_hidden(model, tok, [p + " 2" for p in oP("U_bright")], L).mean(0))}
    log["cos"] |= {f"{p}|{q}": float(d_pos[p] @ D[q]) for p in d_pos for q in D}
    log["cos"]["pos_82|pos_U82"] = float(d_pos["pos_82"] @ d_pos["pos_U82"])
    np.save(out / f"s1n_dirs_{tag}.npy", np.stack([D[k] for k in dirs] + [d_pos[k] for k in d_pos]))
    log["dir_names"] = dirs + list(d_pos)
    pos_task = {"pos_82": "S", "pos_sev": "S", "pos_U82": "U_bright"}
    Reven = {t: Rating(tok, eI(t)) for t in ("S", "U_bright")}
    base_even = {t: R.score(model, steer, None) for t, R in Reven.items()}

    def delta_pos(v, t):
        return float(np.mean(Reven[t].score(model, steer, alpha * v) - base_even[t]))

    res = log.setdefault("readout", {}).setdefault("R9n", {})
    res["effects"] = {nm: dict(zip(COLS, effects(D[nm]).tolist())) for nm in dirs}
    res["pos_even"] = {k: delta_pos(v, pos_task[k]) for k, v in d_pos.items()}
    save()

    # positive controls first (gate), then the paired null, then the others
    for nm, v in d_pos.items():
        tv = float(np.var(Ac @ v))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, a.n_pos)
        z = np.asarray([delta_pos(r, pos_task[nm]) for r in rs])
        np.save(out / f"s1n_null_{nm}_{tag}.npy", z)
        res.setdefault("null", {})[nm] = {"n": a.n_pos, "task": pos_task[nm],
                                          "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv),
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
    np.save(out / f"s1n_pair_cos_{tag}.npy", got)
    save()
    z1 = np.stack([effects(v) for v in r1])
    np.save(out / f"s1n_null_pair_neg_{tag}.npy", z1)
    save()
    z2 = np.stack([effects(v) for v in r2])
    np.save(out / f"s1n_null_pair_neu_{tag}.npy", z2)
    save()

    for nm in ("int_pos", "pc1_neg"):
        tv = float(np.var(Ac @ D[nm]))
        rs = fast_var_matched(np.random.default_rng(SEED[nm]), evals, evecs, len(Ac), tv, a.n_null_minor)
        z = np.stack([effects(v) for v in rs])
        np.save(out / f"s1n_null_{nm}_{tag}.npy", z)
        res["null"][nm] = {"n": a.n_null_minor, "var_check": float(np.median([np.var(Ac @ r) for r in rs]) / tv)}
        save()

    log["t_total_s"] = time.time() - log["t0"]
    save()
    print(json.dumps({k: log.get(k) for k in ("cos_with_1m", "manipulation_max_abs_diff_1m")}, indent=1))
    print(json.dumps({k: v for k, v in log["checks"].items() if k != "base_E"}, indent=1))
    print(json.dumps(res["effects"], indent=1))


if __name__ == "__main__":
    main()
