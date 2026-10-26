"""Stage 1j pre-registration probe 2 (no steering; baseline only): does each readout track a severity manipulation?
Rating formats (forward scale only; readout_R9s.csv): cont, qa, chat (qa question as a chat user turn) -> E[digit].
Option readouts B2, B2n, B2x, B2m: the mild/severe sentence is inserted after the scenario (same split as
build_r9s.py), score = strict-option log-odds.
Per readout: severity effect = mean over the 100 items of score(severe) - score(mild), the fraction of items with a
positive difference, and the effect divided by the SD of the item-level differences. Ratings also report the object
check (Spearman of E[digit] with log weight over the 100 object items).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_light_round import load_csv
from s1c_word_vs_situation import Readout
from s1j_format_probe import _NoSteer, digit_probs

FRAMES_B2 = (" The response chosen by the responsible official was",
             " According to the written policy, this case calls for")
SEV = {"mild": "The person apologized right away and had a clean record.",
       "severe": "The person had done this many times before, despite several written warnings."}


def with_sev(rows, sev):
    out = []
    for x in rows:
        f = [f for f in FRAMES_B2 if x["prompt"].endswith(f)][0]
        out.append(x | {"prompt": x["prompt"][: -len(f)] + " " + SEV[sev] + f})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.stim)
    R = load_csv(S / "readout_R9s.csv")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}

    def chat(q):
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **kw)

    mild = [x for x in R if x["sev"] == "mild"]
    sev = [x for x in R if x["sev"] == "severe"]
    obj = [x for x in R if x["task"] == "O"]
    assert [x["scen"] for x in mild] == [x["scen"] for x in sev]
    wkg = np.log10([float(x["weight_kg"]) for x in obj])
    fmts = {"cont": (lambda x: x["prompt_cont"], " "), "qa": (lambda x: x["prompt_qa"] + "\nAnswer:", " ")}
    if tok.chat_template:
        fmts["chat"] = (lambda x: chat(x["prompt_qa"]), "")
    res = {}

    def summ(d):
        return {"effect": float(d.mean()), "frac_pos": float((d > 0).mean()), "effect_over_sd": float(d.mean() / d.std(ddof=1))}

    for name, (f, sp) in fmts.items():
        conts = [f"{sp}{k}" for k in range(1, 10)]
        e = {k: digit_probs(model, tok, [f(x) for x in rows], conts) @ np.arange(1, 10)
             for k, rows in (("mild", mild), ("severe", sev), ("obj", obj))}
        res[f"rating_{name}"] = summ(e["severe"] - e["mild"]) | {
            "object_rho_logweight": float(spearmanr(e["obj"], wkg)[0]), "mean_E_mild": float(e["mild"].mean()),
            "mean_E_severe": float(e["severe"].mean()), "mean_E_obj": float(e["obj"].mean())}
        print(name, res[f"rating_{name}"], flush=True)
    B = load_csv(S / "readout_B.csv")
    opt = {"B2": [x for x in B if x["readout"] == "B2"], "B2n": load_csv(S / "readout_B2n.csv"),
           "B2x": load_csv(S / "readout_B2x.csv"), "B2m": load_csv(S / "readout_B2m.csv")}
    for name, rows in opt.items():
        sc = {s: Readout(tok, with_sev(rows, s)).score(model, _NoSteer(), None) for s in SEV}
        res[f"option_{name}"] = summ(sc["severe"] - sc["mild"])
        print(name, res[f"option_{name}"], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"s1j_probe2_{a.model.split('/')[-1]}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
