"""Stage 1n readout probe (no steering; before any Stage 1n registration).
For each question of readout_R9n_probe.csv: the baseline expected digit E of the mild and severe versions of the 60
Stage 1k scenarios. Reports
  severity effect (severe - mild) and its sign fraction,
  item consistency: Spearman of E(mild) with E(severe) over scenarios (does the question follow a scenario property),
  spread: SD of the scenario mean over scenarios,
  Spearman with S over all 120 texts,
  for C: Spearman with the designed location.
S, C and L must reproduce the Stage 1m baseline (same items, same code path); the max difference is reported.
Probe 2: --csv readout_R9n_probe2.csv --tasks S,U_size,U_bright,U_formal,U_unusual --suffix _p2 (no C/L reproduction)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, load_csv
from s1j_rating import Rating

TASKS = ("S", "C", "L", "HARSH", "MILD", "U_crowd", "U_noise", "U_old", "U_fam")
K9 = np.arange(1, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--res1m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.33)
    ap.add_argument("--csv", default="readout_R9n_probe.csv")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    tasks = tuple(a.tasks.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_anger_d{a.depth}"
    t0 = time.time()
    R = load_csv(Path(a.stim) / a.csv)
    for x in R:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    get = lambda t, s: [x for x in R if x["task"] == t and x["sev"] == s]
    blocks = {(t, s): get(t, s) for t in tasks for s in ("mild", "severe")}
    ref = [x["scen"] for x in blocks[("S", "mild")]]
    assert all([x["scen"] for x in b] == ref for b in blocks.values())
    n = len(ref)
    loc = np.array([int(x["loc"]) for x in blocks[("S", "mild")]])

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    L = int(round(a.depth * cfg.num_hidden_layers))
    steer = Steer(decoder_layers(model)[L - 1])
    order = [(t, s) for t in tasks for s in ("mild", "severe")]
    E = Rating(tok, sum((blocks[k] for k in order), [])).score(model, steer, None)
    e = {k: E[i * n:(i + 1) * n] for i, k in enumerate(order)}

    allS = np.r_[e[("S", "mild")], e[("S", "severe")]]
    res = {}
    for t in tasks:
        m, s = e[(t, "mild")], e[(t, "severe")]
        d = s - m
        res[t] = {"mean_mild": float(m.mean()), "mean_severe": float(s.mean()),
                  "severity_effect": float(d.mean()), "frac_pos": float((d > 0).mean()),
                  "frac_neg": float((d < 0).mean()), "rho_mild_severe": float(spearmanr(m, s)[0]),
                  "sd_scenario": float(np.std((m + s) / 2, ddof=1)),
                  "rho_with_S": float(spearmanr(np.r_[m, s], allS)[0])}
    if "C" in res:
        res["C"]["rho_location"] = float(spearmanr((e[("C", "mild")] + e[("C", "severe")]) / 2, loc)[0])

    # reproduction of the Stage 1m baseline for S, C, L
    j = json.load(open(Path(a.res1m) / f"s1m_{tag}.json"))
    b1m = np.asarray(j["checks"]["base_E"])
    if all(t in tasks for t in ("S", "C", "L")):
        mine = np.r_[tuple(e[(t, s)] for t in ("S", "C", "L") for s in ("mild", "severe"))]
        repro = float(np.max(np.abs(mine - b1m)))
    else:
        repro = float(np.max(np.abs(np.r_[e[("S", "mild")], e[("S", "severe")]] - b1m[:2 * n])))
    log = {"args": vars(a), "layer": L, "tasks": res, "repro_1m_max_abs_diff": repro,
           "base_E": {f"{t}|{s}": e[(t, s)].tolist() for t, s in order}, "t_total_s": time.time() - t0}
    (out / f"s1n_probe{a.suffix}_{tag}.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"repro_1m_max_abs_diff": log["repro_1m_max_abs_diff"]}
                     | {t: {k: round(v, 2) for k, v in r.items()} for t, r in res.items()}))


if __name__ == "__main__":
    main()
