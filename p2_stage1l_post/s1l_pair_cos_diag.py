"""Stage 1l diagnostic (not registered): per-draw cosine of the paired null.
Rebuilds the covariance, pc1_neg and pc1_pos exactly as s1l_rating.py (same model load, texts and layer), reruns
paired_var_matched with the registered seed, and saves the per-draw cos(r_neg, r_pos). Row i matches row i of
s1l_null_pair_neg/pos. Checks that the rebuilt directions equal the saved ones (s1l_dirs_*.npy). No steering."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1_light_round import decoder_layers, last_hidden, load_csv, unit  # noqa: E402
from s1l_rating import COV_SETS, SEED, SETS, paired_var_matched  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--n_pair", type=int, default=1000)
    a = ap.parse_args()
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"

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

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    assert decoder_layers(model)[L - 1] is not None
    with torch.no_grad():
        H = {k: last_hidden(model, tok, v, L) for k, v in texts.items()}
        H_ctx = last_hidden(model, tok, C_ctx, L)
    Ac = np.concatenate([H_ctx] + [H[k] for k in H if k[1:] in ("C",) + COV_SETS])
    Ac = Ac - Ac.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Ac.T))

    m = {k: v.mean(0) for k, v in H.items()}
    W = {x: unit((m[f"S{x}"] + m[f"N{x}"]) / 2 - (m["SC"] + m["NC"]) / 2) for x in SETS}

    def pc1(names):
        M = np.stack([W[x] for x in names])
        pc = np.linalg.svd(M, full_matrices=False)[2][0]
        return unit(pc if pc @ M.mean(0) > 0 else -pc)

    neg, pos = pc1(("anger", "eval", "fear")), pc1(("joy", "grat", "poseval"))
    saved = np.load(Path(a.res) / f"s1l_dirs_{tag}.npy")
    tv1, tv2 = float(np.var(Ac @ neg)), float(np.var(Ac @ pos))
    c12 = float(neg @ pos)
    _, _, got = paired_var_matched(np.random.default_rng(SEED["pair"]), evals, evecs, len(Ac), tv1, tv2, c12,
                                   a.n_pair)
    j = json.load(open(Path(a.res) / f"s1l_{tag}.json"))["readout"]["R9k"]["null"]["pair"]
    err = got - c12
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"s1l_pair_cos_{tag}.npy", got)
    rep = {"model": a.model, "cos_target": c12, "cos_target_saved": j["cos_target"],
           "max_abs_cos_rebuilt_vs_saved_dirs": float(max(abs(1 - abs(float(neg @ v))) for v in saved[:1])),
           "median_got_rebuilt": float(np.median(got)), "median_got_saved": j["cos_got_median"],
           "max_abs_err_rebuilt": float(np.max(np.abs(err))), "max_abs_err_saved": j["cos_got_max_abs_err"],
           "n_err_gt_0.01": int(np.sum(np.abs(err) > 0.01)), "n_err_gt_0.001": int(np.sum(np.abs(err) > 0.001)),
           "err_sign_of_misses": float(np.mean(np.sign(err[np.abs(err) > 0.01]))) if np.any(np.abs(err) > 0.01) else None,
           "var_ratio_tv2_over_tv1": tv2 / tv1}
    (out / f"s1l_pair_cos_{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
