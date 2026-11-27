"""P3-K prereg v3.3 eligibility test (02_設計/06_課題生成規則_v3.3_20261127.md §6). Run ONCE; failing tasks are excluded.

Input : stimuli_v3/p3k_pilot.csv (10 pilot scenarios per task, strong and weak versions, readouts J and CTRL only).
Rule  : a task is eligible iff J severity effect (mean strong - weak of the expected digit) > 1 and
        |Spearman rho(CTRL score, J score)| < .3 over the 20 pilot units. Nothing else is computed: no JR, no choice
        readout, no gradient, no steering.
Output: <out>/p3k_eligibility.json (per task: statistics, eligible) and the list of eligible tasks.
Usage : python p3k_pilot.py --model M --revision R --pilot stimuli_v3/p3k_pilot.csv --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

K9 = np.arange(1, 10)
J_EFFECT_MIN, CTRL_RHO_MAX = 1.0, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fo = out / "p3k_eligibility.json"
    assert not fo.exists(), "the eligibility test is run once only (prereg v3.3)"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    rows = list(csv.DictReader(open(a.pilot, encoding="utf-8")))
    seqs, cids = [], None
    for r in rows:
        full = [tok(r["prompt_qa"] + "\nAnswer:" + f" {k}")["input_ids"] for k in K9]
        pre = full[0][:-1]
        assert all(f[:-1] == pre for f in full)
        seqs.append(pre)
        c = [f[-1] for f in full]
        assert cids is None or c == cids
        cids = c
    order = np.argsort([len(s) for s in seqs], kind="stable")
    E = np.zeros(len(rows))
    ct = torch.tensor(cids).cuda()
    with torch.no_grad():
        for b in range(0, len(order), 64):
            pos = order[b:b + 64]
            T = max(len(seqs[i]) for i in pos)
            ids = torch.full((len(pos), T), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(pos), T), dtype=torch.long)
            for j, i in enumerate(pos):
                ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
                mask[j, :len(seqs[i])] = 1
            last = mask.sum(1) - 1
            lg = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits[torch.arange(len(pos)), last.cuda()]
            p = torch.softmax(lg.float()[:, ct], -1).cpu().numpy()
            E[pos] = p @ K9
    res = {"rule": {"J_effect_min": J_EFFECT_MIN, "ctrl_rho_max": CTRL_RHO_MAX},
           "pilot_sha256": hashlib.sha256(open(a.pilot, "rb").read()).hexdigest(), "tasks": {}}
    tasks = sorted({r["ptask"] for r in rows}, key=lambda t: (t[0] != "V", int(t[1:])))
    for t in tasks:
        sc = {}
        for ro in ("J", "CTRL"):
            sc[ro] = {(r["sev"], r["scen"]): E[i] for i, r in enumerate(rows) if r["ptask"] == t and r["readout"] == ro}
        keys = sorted(sc["J"])
        assert keys == sorted(sc["CTRL"]) and len(keys) == 20, (t, len(keys))
        scen = sorted({s for _, s in keys})
        d = np.array([sc["J"][("strong", s)] - sc["J"][("weak", s)] for s in scen])
        rho = float(spearmanr([sc["CTRL"][k] for k in keys], [sc["J"][k] for k in keys])[0])
        rho = 0.0 if np.isnan(rho) else rho
        ok = bool(d.mean() > J_EFFECT_MIN and abs(rho) < CTRL_RHO_MAX)
        res["tasks"][t] = {"J_effect": float(d.mean()), "J_frac_pos": float((d > 0).mean()), "rho_ctrl_J": rho,
                           "ctrl_sd": float(np.std([sc["CTRL"][k] for k in keys])), "eligible": ok}
    res["eligible"] = [t for t in tasks if res["tasks"][t]["eligible"]]
    res["excluded"] = [t for t in tasks if not res["tasks"][t]["eligible"]]
    fo.write_text(json.dumps(res, indent=1))
    print(json.dumps({"n_eligible": len(res["eligible"]), "excluded": res["excluded"]}))


if __name__ == "__main__":
    main()
