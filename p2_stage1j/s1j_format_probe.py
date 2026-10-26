"""Stage 1j pre-registration probe: which 1-9 rating format is a valid readout? No steering; baseline only.

For each format, every (task, scale, item) is scored as p(digit = k) for k = 1..9, computed as the log-probability of
the continuation text of each digit (full-sequence scoring, so any tokenisation works), renormalised over the nine.
Formats:
  cont: the readout_R9.csv prompt ("... The rating was") + " k"
  qa:   "Scenario/Object ... Question: ... Answer with one digit.\nAnswer:" + " k"
  chat: the qa question as a user turn in the model's chat template (generation prompt added) + "k"
Checks per format (same as the registered-to-be gate): reversal_S, reversal_O (Spearman of f and reversed r scores),
judgment_rho_B2_within_pair (Spearman of S combined score with pair-demeaned baseline B2), object_rho_logweight.
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

S_Q = "On a scale from 1 ({lo}) to 9 ({hi}), how severe a sanction does this case deserve? Answer with one digit."
O_Q = "On a scale from 1 ({lo}) to 9 ({hi}), how heavy is it? Answer with one digit."
ANCH = {"S": ("no sanction at all", "the most severe sanction"), "O": ("extremely light", "extremely heavy")}


@torch.no_grad()
def digit_probs(model, tok, prompts, conts, bs=64):
    """prompts: list[str]; conts: list of 9 continuation strings. Returns (len(prompts), 9) probabilities."""
    seqs, spans = [], []
    for p in prompts:
        n_p = len(tok(p, add_special_tokens=False)["input_ids"]) if p.startswith(tok.bos_token or "\0") else \
            len(tok(p)["input_ids"])
        for c in conts:
            ids = tok(p + c, add_special_tokens=not p.startswith(tok.bos_token or "\0"))["input_ids"]
            seqs.append(ids)
            spans.append((min(max(n_p, 1), len(ids) - 1), len(ids)))
    T = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
        mask[j, : len(s)] = 1
    ids, mask = ids.cuda(), mask.cuda()
    lps = []
    for b in range(0, len(seqs), bs):
        lp = torch.log_softmax(model(input_ids=ids[b:b + bs], attention_mask=mask[b:b + bs]).logits.float(), -1)
        for j in range(lp.shape[0]):
            s, e = spans[b + j]
            lps.append(lp[j, s - 1:e - 1].gather(1, ids[b + j, s:e][:, None]).sum().item())
    lps = np.array(lps).reshape(-1, len(conts))
    p = np.exp(lps - lps.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.stim)
    R9 = load_csv(S / "readout_R9.csv")
    R_B2 = [x for x in load_csv(S / "readout_B.csv") if x["readout"] == "B2"]
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()

    def subject(x):  # scenario text or "Object: ..." text, recovered from the cont prompt
        p = x["prompt"]
        if x["task"] == "S":
            return "Scenario: " + p.split(" The responsible official rated")[0].split(" According to the written policy")[0]
        return "Object: " + p.split(".")[0].replace("Consider ", "")

    def qa_text(x):
        lo, hi = ANCH[x["task"]] if x["scale"] == "f" else ANCH[x["task"]][::-1]
        q = (S_Q if x["task"] == "S" else O_Q).format(lo=lo, hi=hi)
        return f"{subject(x)}\nQuestion: {q}"

    def chat(x):
        kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}
        try:
            return tok.apply_chat_template([{"role": "user", "content": qa_text(x)}], tokenize=False,
                                           add_generation_prompt=True, **kw)
        except Exception:
            return None

    fmts = {"cont": ([x["prompt"] for x in R9], [f" {k}" for k in range(1, 10)]),
            "qa": ([qa_text(x) + "\nAnswer:" for x in R9], [f" {k}" for k in range(1, 10)])}
    if tok.chat_template:
        fmts["chat"] = ([chat(x) for x in R9], [f"{k}" for k in range(1, 10)])
    base_B2 = Readout(tok, R_B2).score(model, _NoSteer(), None)
    pair = np.array([x["option_a"] if x["target_option"] == "a" else x["option_b"] for x in R_B2])
    dm = base_B2 - np.array([base_B2[pair == p].mean() for p in pair])
    idx = {(t, s): [i for i, x in enumerate(R9) if x["task"] == t and x["scale"] == s] for t in "SO" for s in "fr"}
    wkg = np.log10([float(R9[i]["weight_kg"]) for i in idx["O", "f"]])
    rho = lambda u, v: float(spearmanr(u, v)[0])
    res = {}
    for name, (prompts, conts) in fmts.items():
        P = digit_probs(model, tok, prompts, conts)
        e = P @ np.arange(1, 10)
        sc = {k: (10 - e[v] if k[1] == "r" else e[v]) for k, v in idx.items()}
        s_comb, o_comb = (sc["S", "f"] + sc["S", "r"]) / 2, (sc["O", "f"] + sc["O", "r"]) / 2
        res[name] = {"reversal_S": rho(sc["S", "f"], sc["S", "r"]), "reversal_O": rho(sc["O", "f"], sc["O", "r"]),
                     "judgment_rho_B2_within_pair": rho(s_comb, dm), "judgment_rho_f_only": rho(sc["S", "f"], dm),
                     "object_rho_logweight": rho(o_comb, wkg), "object_rho_f_only": rho(sc["O", "f"], wkg),
                     "mean_E": {f"{t}{s}": float(e[v].mean()) for (t, s), v in idx.items()},
                     "mean_maxprob": float(P.max(1).mean()), "example_prompt": prompts[idx["S", "r"][0]]}
        print(name, {k: v for k, v in res[name].items() if k != "example_prompt"}, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"s1j_probe_{a.model.split('/')[-1]}.json").write_text(json.dumps(res, indent=1))


class _NoSteer:
    v = None


if __name__ == "__main__":
    main()
