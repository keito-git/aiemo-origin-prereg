"""Stage 1u pilot (no steering): can a fixed few-shot format prefix make the readouts of the base models valid?
Stage 1t smoke: Llama-3.1-8B and OLMo-2-1124-7B failed the Stage 1o entry rule because no choice-format (CHO)
readout passed (severity effects 0.2-0.7). Here every readout prompt of readout_R9o_main.csv is preceded by
demonstrations of the answer format only. The demonstrations are about neutral, factual, non-social topics (no
sanction, harm, norm or emotion content), so they show how to answer, not what the target judgment is.
Variants: F0 (no prefix; must reproduce the Stage 1t smoke base scores), F4 (4 demonstrations), F8 (8 demonstrations).
Only base scores and the registered validity function (s1o_pooled_tests.validity) are computed; no direction, no
steering, no effect, so no outcome of the excess analysis is seen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers, load_csv
from s1o_pooled_tests import validity
from s1o_rating import CAT4, DIGIT, JUDG, CTRL, K9
from s1p_rating import build_readouts

# Format demonstrations; letters balanced over the 4-choice demos, digits balanced around the scale middle.
D_CAT = [
    ("A student measured the length of a pencil with a ruler.", "Which unit fits this measurement best?",
     ["kilometers", "centimeters", "tons", "liters"], "B"),
    ("A family kept a bottle of milk cold overnight.", "Where was the milk most likely kept?",
     ["in an oven", "on a bookshelf", "in a garden shed", "in a refrigerator"], "D"),
    ("A worker painted the walls of a small office.", "Which tool did the worker most likely use?",
     ["a hammer", "a microscope", "a paint roller", "a telescope"], "C"),
    ("A farmer collected eggs from the barn in the morning.", "Which animal laid the eggs?",
     ["a hen", "a cow", "a horse", "a goat"], "A"),
]
D_PAIR = [
    ("A traveler needed to cross a wide lake.", "Which is the more fitting next step?",
     ["taking a boat", "riding a bicycle"], "A"),
    ("A cook noticed that the soup had gone cold.", "Which is the more fitting next step?",
     ["putting it in the freezer", "heating it on the stove"], "B"),
]
D_DIG = [
    ("A cook boiled a pot of water for pasta.",
     "On a scale from 1 (very cold) to 9 (extremely hot), how hot was the water?", "9"),
    ("A child picked up a single feather from the ground.",
     "On a scale from 1 (extremely light) to 9 (extremely heavy), how heavy was the object?", "1"),
]


def demo_cat(s, q, opts, ans):
    o = "\n".join(f"{l}. {t}" for l, t in zip("ABCD", opts))
    return f"Scenario: {s}\nQuestion: {q}\n{o}\nAnswer with one letter.\nAnswer: {ans}"


def demo_dig(s, q, ans):
    return f"Scenario: {s}\nQuestion: {q} Answer with one digit.\nAnswer: {ans}"


WRONG = {"A": "C", "B": "D", "C": "A", "D": "B", "9": "1", "1": "9"}   # 2-choice: A<->B below


def prefix_wrong() -> str:
    """F8 with every demonstrated answer wrong (R3 control, exploratory): letters permuted, PAIR swapped, 9<->1."""
    ds = []
    for d in prefix(8).split("\n\n")[:-1]:
        head, ans = d.rsplit("Answer: ", 1)
        n_opt = sum(head.count(f"\n{l}. ") for l in "ABCD")
        w = {"A": "B", "B": "A"}[ans] if n_opt == 2 else WRONG[ans]
        ds.append(f"{head}Answer: {w}")
    return "\n\n".join(ds) + "\n\n"


def prefix(k: int) -> str:
    if k == -8:
        return prefix_wrong()
    if k == 0:
        return ""
    if k == 4:
        ds = [demo_cat(*D_CAT[0]), demo_cat(*D_PAIR[0]), demo_dig(*D_DIG[0]), demo_dig(*D_DIG[1])]
    elif k == 8:
        ds = ([demo_cat(*D_CAT[0]), demo_cat(*D_PAIR[0]), demo_dig(*D_DIG[0]), demo_cat(*D_CAT[1]),
               demo_cat(*D_CAT[2]), demo_dig(*D_DIG[1]), demo_cat(*D_PAIR[1]), demo_cat(*D_CAT[3])])
    else:
        raise ValueError(k)
    return "\n\n".join(ds) + "\n\n"


def base_scores(model, steer, RO):
    """Identical to the base-score block of s1o_rating.main."""
    base = {t: r.logp(model, steer, None) for t, r in RO.items()}
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
    return sc


def entry(v):
    j = [t for t in JUDG if v[t]]
    return {"valid_judg": j, "valid_ctrl": [t for t in CTRL if v[t]],
            "entry_readouts": len(j) >= 3 and any(t in j for t in ("S", "HARSH", "L"))
            and any(t in j for t in ("CAT_S", "CAT_S2", "PAIR")) and any(v[t] for t in CTRL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", default="0,4,8")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    steer = Steer(decoder_layers(model)[int(round(0.33 * cfg.num_hidden_layers)) - 1])
    log = {"args": vars(a), "t0": time.time(), "res": {}}
    for k in [int(x) for x in a.k.split(",")]:
        R9 = load_csv(Path(a.stim) / "readout_R9o_main.csv")
        pf = prefix(k)
        for x in R9:
            x["prompt"] = pf + x["prompt_qa"] + "\nAnswer:"
        RO, _ = build_readouts(tok, R9)
        sc = base_scores(model, steer, RO)
        s, v = validity(sc)
        log["res"][f"F{k}"] = {"prefix": pf, "summary": s, "valid": v, "entry": entry(v), "scores": sc}
        print(k, {t: round(s[t]["severity_effect"], 2) for t in JUDG}, entry(v), flush=True)
    log["t_total_s"] = time.time() - log["t0"]
    (out / f"s1u_pilot_{a.model.split('/')[-1]}.json").write_text(json.dumps(log, indent=1, default=float))


if __name__ == "__main__":
    main()
