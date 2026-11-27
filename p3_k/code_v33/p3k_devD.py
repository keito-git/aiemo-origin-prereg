"""P3-K prereg v3.2, stage D (measurement development; plan 00_研究計画/21_事前登録v3.2設計_測定開発と新課題群_20261127.md).

The 28 tasks of v3.1 (V1-V16, N1-N12) become the measurement-development set. This script computes ONLY no-steering
validity (P2 §4.2 criteria, p3k_common.validity_readout) and the task-entry rule for readout-design variants and label
sets. No gradient, no steering, no positive control, no emotion direction is used, so nothing here touches K1-K3.

Design variants (stimulus strength unchanged; prompts built from ../p3k_items_{v,n}.py with the v1 builder helpers):
  CAT / CTRL_CAT presentation   cyc = 4 cyclic rotations (v3.1)   ord = ascending and descending rank order
  PAIR rank pairs               adj = 12, 23, 34 (v3.1)            far = 13, 24, 14          (both orders kept)
  D0 = cyc+adj, D1 = ord+adj, D2 = cyc+far, D12 = ord+far.
Label sets: 1 = A B C D, 2 = Q X Z J (primary); Ka = H K R W, Kb = G N P T, 3 = psi xi zeta theta (K3 candidates).
Per (task, label set) the label quad of each scenario pair is drawn exactly as in build_p3k_stimuli (Latin square,
balanced, seeded), so D0 with sets 1 and 2 reproduces the v3.1 prompts byte for byte (asserted against stimuli_v1).

Usage
  GPU:  python p3k_devD.py run --model M --revision R --out DIR [--tasks ...] [--sets 1,2,Ka,Kb,3]
  CPU:  python p3k_devD.py select --out DIR      (applies the pre-specified selection rules, plan §1.2 and §3.2)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):            # local layout: code/pipeline_v0 + code; server layout: one flat folder
    if (p / "build_p3k_stimuli.py").exists():
        sys.path.insert(0, str(p))
import build_p3k_stimuli as B             # noqa: E402  (rating_prompt, choice_tpl, render, split_slot, ...)
from p3k_items_n import TASKS_N           # noqa: E402
from p3k_items_v import TASKS_V           # noqa: E402
from p3k_common import K9, W9, rot_cons, summ, task_entry  # noqa: E402

TASKS = tuple([f"V{i}" for i in range(1, 17)] + [f"N{i}" for i in range(1, 13)])
LABEL_SETS = {"1": ["A", "B", "C", "D"], "2": ["Q", "X", "Z", "J"], "Ka": ["H", "K", "R", "W"],
              "Kb": ["G", "N", "P", "T"], "3": ["ψ", "ξ", "ζ", "θ"]}
PRIMARY = ("1", "2")
K3_CAND = ("Ka", "Kb", "3")
PRES = {"cyc": [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]], "ord": [[0, 1, 2, 3], [3, 2, 1, 0]]}
PAIRS = {"adj": [(0, 1), (1, 2), (2, 3)], "far": [(0, 2), (1, 3), (0, 3)]}
VARIANTS = {"D0": ("cyc", "adj"), "D1": ("ord", "adj"), "D2": ("cyc", "far"), "D12": ("ord", "far")}
VAR_ORDER = ("D0", "D1", "D2", "D12")      # tie-break: fewer changes first (plan §1.2)
N_MIN_KEEP_P2 = 21                         # 75% of 28: keep the P2 thresholds if the best variant reaches it
CAT_EFFECT_HARMONIZED = 0.375              # 12.5% of the 1-4 range (= J's 1 on the 1-9 range), plan §1.2 step 4
SEVS = ("weak", "strong")


def quad_index(ptask: str, set_key: str) -> list[int]:
    """Same seeded balanced assignment as build_p3k_stimuli (integer keys for sets 1-3 reproduce v3.1)."""
    key = int(set_key) if set_key.isdigit() else set_key
    return B.balanced_assignment(ptask, key, 30)


def items_for(ptask: str, readout: str, set_key: str, pres: str, pairs: str) -> list[dict]:
    """Items in canonical order: presentation x severity (weak, strong) x scenario. Each item: prompt, choices
    (answer strings in position order), w (score weights), scen."""
    T = {**TASKS_V, **TASKS_N}[ptask]
    labs = LABEL_SETS[set_key]
    quads = [[labs[(k + s) % 4] for k in range(4)] for s in range(4)]
    qi = quad_index(ptask, set_key)
    vers = []
    for i, (text, _ctrl) in enumerate(T["items"], 1):
        strong, weak, *_ = B.split_slot(text)
        vers.append((f"{ptask}_{i:02d}", {"strong": strong, "weak": weak}, quads[qi[i - 1]]))
    out = []
    if readout in ("J", "JR", "CTRL"):
        lo, hi, q = T[readout]
        for s in SEVS:
            for scen, v, _ in vers:
                out.append({"scen": scen, "prompt": B.rating_prompt(v[s], lo, hi, q), "rot": "",
                            "choices": [str(k) for k in K9], "w": W9})
        return out
    if readout in ("CAT", "CTRL_CAT"):
        q, opts = T[readout]
        for r, perm in enumerate(PRES[pres]):
            for s in SEVS:
                for scen, v, quad in vers:
                    tpl = B.choice_tpl(v[s], q, [opts[j] for j in perm])
                    rk = np.array([j + 1 for j in perm], float)
                    out.append({"scen": scen, "prompt": B.render(tpl, quad), "rot": str(r), "choices": quad[:4],
                                "w": (rk - 2.5) / 5.0})
        return out
    assert readout == "PAIR"
    q, opts = T["PAIR_Q"], T["CAT"][1]
    for a, b in PAIRS[pairs]:
        for tag, (x, y) in (("a", (a, b)), ("b", (b, a))):
            for s in SEVS:
                for scen, v, quad in vers:
                    tpl = B.choice_tpl(v[s], q, [opts[x], opts[y]])
                    rk = np.array([x + 1, y + 1], float)
                    out.append({"scen": scen, "prompt": B.render(tpl, quad), "rot": f"{a + 1}{b + 1}{tag}",
                                "choices": quad[:2], "w": np.where(rk == rk.max(), 1.0, -1.0)})
    return out


def scores(readout: str, logp: np.ndarray, items: list[dict]) -> dict:
    """p3k_common.base_scores generalized to any number of presentations (same definitions)."""
    n_unit = 60
    if readout in ("J", "JR", "CTRL"):
        return {"score": (np.exp(logp) @ K9).tolist()}
    W = np.stack([x["w"] for x in items])
    if readout in ("CAT", "CTRL_CAT"):
        E = (np.exp(logp) * (W * 5.0 + 2.5)).sum(1).reshape(-1, n_unit)
        return {"score": E.mean(0).tolist(), "by_rot": E.tolist()}
    lo = (logp * W).sum(1).reshape(-1, n_unit)
    return {"score": lo.mean(0).tolist(), "by_rot": lo.tolist()}


def validity(readout: str, sc: dict, ref_J, cat_thr: float = 1.0) -> tuple[dict, bool]:
    """p3k_common.validity_readout with the CAT effect threshold as a parameter (1.0 = P2)."""
    s = summ(sc["score"], ref_J)
    if readout in ("CAT", "CTRL_CAT"):
        s["rot_consistency"] = rot_cons(sc["by_rot"])
    if readout == "PAIR":
        from scipy.stats import spearmanr
        P = np.asarray(sc["by_rot"])
        s["order_consistency"] = float(np.mean([spearmanr(P[2 * k], P[2 * k + 1])[0] for k in range(3)]))
    v = {"J": lambda: s["severity_effect"] > 1 and s["frac_pos"] >= .8,
         "JR": lambda: s["severity_effect"] < -.5 and s["frac_neg"] >= .8 and s["rho_with_S"] < -.5,
         "CAT": lambda: s["severity_effect"] > cat_thr and s["rot_consistency"] > .8 and s["rho_with_S"] > .7,
         "PAIR": lambda: s["severity_effect"] > 1 and s["frac_pos"] >= .8 and s["rho_with_S"] > .7,
         "CTRL": lambda: abs(s["severity_effect"]) < 1.2 and s["rho_mild_severe"] > .5 and abs(s["rho_with_S"]) < .3,
         "CTRL_CAT": lambda: (abs(s["severity_effect"]) < .5 and s["rot_consistency"] > .5
                              and abs(s["rho_with_S"]) < .3)}[readout]()
    return s, bool(v)


def check_reproduction(stim: Path, tasks: list[str]) -> None:
    """D0 prompts for sets 1, 2 and 3 must equal the stimuli_v1 rows of v3.1 (all readouts, every item)."""
    for sk in ("1", "2", "3"):
        rows = list(csv.DictReader(open(stim / f"p3k_readouts_map{sk}.csv", encoding="utf-8")))
        ref = {(r["ptask"], r["readout"], r["rot"], r["sev"], r["scen"]): r["prompt_qa"] for r in rows}
        n = 0
        for t in tasks:
            for ro in ("J", "JR", "CTRL", "CAT", "CTRL_CAT", "PAIR"):
                for k, it in enumerate(items_for(t, ro, sk, "cyc", "adj")):
                    sev = SEVS[(k // 30) % 2]
                    assert ref[(t, ro, it["rot"], sev, it["scen"])] == it["prompt"], ("D0 != v3.1", t, ro, sk, k)
                    n += 1
        print(f"D0_REPRODUCES_V31 set {sk}: {n} prompts", flush=True)


# ------------------------------------------------------------------ GPU
def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    pad = tok.pad_token_id

    @torch.no_grad()
    def logp(items: list[dict], bs: int = 64) -> np.ndarray:
        seqs, cids = [], []
        for it in items:
            p = it["prompt"] + "\nAnswer:"
            full = [tok(p + f" {k}")["input_ids"] for k in it["choices"]]
            pre = full[0][:-1]
            assert all(f[:-1] == pre for f in full), ("answer is not one token after the prefix", it["choices"])
            seqs.append(pre)
            cids.append([f[-1] for f in full])
        order = np.argsort([len(s) for s in seqs], kind="stable")
        P = np.zeros((len(items), len(cids[0])))
        for b in range(0, len(order), bs):
            pos = order[b:b + bs]
            T = max(len(seqs[i]) for i in pos)
            ids = torch.full((len(pos), T), pad, dtype=torch.long)
            mask = torch.zeros((len(pos), T), dtype=torch.long)
            for j, i in enumerate(pos):
                ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
                mask[j, :len(seqs[i])] = 1
            last = mask.sum(1) - 1
            lg = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits[torch.arange(len(pos)), last.cuda()]
            c = torch.tensor([cids[i] for i in pos]).cuda()
            P[pos] = torch.log_softmax(torch.gather(lg.float(), 1, c), -1).cpu().numpy()
        return P

    check_reproduction(Path(a.stim), a.tasks.split(","))

    for t in a.tasks.split(","):
        f = out / f"dev_{t}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = {"task": t, "digit": {}, "label": {}}
        dig = {}
        for ro in ("J", "JR", "CTRL"):
            its = items_for(t, ro, "1", "cyc", "adj")
            dig[ro] = scores(ro, logp(its), its)
        ref_J = dig["J"]["score"]
        res["ref_J"] = ref_J
        for ro in ("J", "JR", "CTRL"):
            s, v = validity(ro, dig[ro], ref_J)
            res["digit"][ro] = {"summ": s, "valid": v}
        for sk in a.sets.split(","):
            for pres in ("cyc", "ord"):
                for ro in ("CAT", "CTRL_CAT"):
                    its = items_for(t, ro, sk, pres, "adj")
                    res["label"][f"{sk}|{ro}|{pres}"] = scores(ro, logp(its), its)
            for pairs in ("adj", "far"):
                its = items_for(t, "PAIR", sk, "cyc", pairs)
                res["label"][f"{sk}|PAIR|{pairs}"] = scores("PAIR", logp(its), its)
        res["t_s"] = time.time() - t0
        f.write_text(json.dumps(res))
        print(t, f"{res['t_s']:.0f}s", flush=True)
    print("DEV_D_RUN_DONE", flush=True)


# ------------------------------------------------------------------ CPU: pre-specified selection
def entry(res: dict, sk: str, var: str, cat_thr: float) -> tuple[bool, dict]:
    pres, pairs = VARIANTS[var]
    valid = {ro: res["digit"][ro]["valid"] for ro in ("J", "JR", "CTRL")}
    summ_ = {}
    for ro, key in (("CAT", f"{sk}|CAT|{pres}"), ("CTRL_CAT", f"{sk}|CTRL_CAT|{pres}"), ("PAIR", f"{sk}|PAIR|{pairs}")):
        s, v = validity(ro, res["label"][key], res["ref_J"], cat_thr)
        valid[ro] = v
        summ_[ro] = s
    return task_entry(valid), {"valid": valid, "summ": summ_}


def select(a):
    out = Path(a.out)
    R = {t: json.load(open(out / f"dev_{t}.json")) for t in TASKS if (out / f"dev_{t}.json").exists()}
    assert len(R) == len(TASKS), f"only {len(R)} of {len(TASKS)} tasks"
    rep = {"n_tasks": len(R), "primary_sets": PRIMARY, "rule": "plan 21 §1.2 and §3.2", "variants": {}}
    for thr_name, thr in (("P2", 1.0), ("harmonized", CAT_EFFECT_HARMONIZED)):
        for var in VAR_ORDER:
            ent = {t: {sk: entry(R[t], sk, var, thr)[0] for sk in LABEL_SETS} for t in R}
            n_both = sum(ent[t]["1"] and ent[t]["2"] for t in R)
            rep["variants"][f"{var}|{thr_name}"] = {"n_both_primary": n_both,
                                                    "n_per_set": {sk: sum(ent[t][sk] for t in R) for sk in LABEL_SETS},
                                                    "tasks_both": sorted(t for t in R if ent[t]["1"] and ent[t]["2"])}
    # step 1-3: best variant under the P2 thresholds (ties -> fewer changes)
    best = max(VAR_ORDER, key=lambda v: (rep["variants"][f"{v}|P2"]["n_both_primary"], -VAR_ORDER.index(v)))
    thr_name = "P2"
    if rep["variants"][f"{best}|P2"]["n_both_primary"] < N_MIN_KEEP_P2:
        # step 4: harmonized CAT threshold; the variant is re-chosen under it with the same rule
        thr_name = "harmonized"
        best = max(VAR_ORDER, key=lambda v: (rep["variants"][f"{v}|harmonized"]["n_both_primary"],
                                             -VAR_ORDER.index(v)))
    thr = 1.0 if thr_name == "P2" else CAT_EFFECT_HARMONIZED
    rep["chosen"] = {"variant": best, "presentation": VARIANTS[best][0], "pairs": VARIANTS[best][1],
                     "cat_threshold": thr_name, "cat_effect_min": thr,
                     "n_both_primary": rep["variants"][f"{best}|{thr_name}"]["n_both_primary"]}
    # K3 mapping gate (§3.2, stage D): median CAT presentation consistency >= .8 and entries >= 0.9 x entries in set 1
    n1 = rep["variants"][f"{best}|{thr_name}"]["n_per_set"]["1"]
    k3 = {}
    for sk in K3_CAND:
        rc = [entry(R[t], sk, best, thr)[1]["summ"]["CAT"]["rot_consistency"] for t in R]
        n = rep["variants"][f"{best}|{thr_name}"]["n_per_set"][sk]
        k3[sk] = {"labels": LABEL_SETS[sk], "median_cat_consistency": float(np.median(rc)), "n_enter": n,
                  "n_enter_set1": n1, "admissible": bool(np.median(rc) >= .8 and n >= 0.9 * n1)}
    rep["k3_candidates"] = k3
    rep["k3_maps_frozen"] = [sk for sk in K3_CAND if k3[sk]["admissible"]]
    rep["readout_medians"] = {
        f"{sk}|{ro}": {k: float(np.nanmedian([entry(R[t], sk, best, thr)[1]["summ"][ro][k] for t in R]))
                       for k in ("severity_effect", "rho_with_S")}
        for sk in LABEL_SETS for ro in ("CAT", "PAIR")}
    (out / "devD_selection.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({"chosen": rep["chosen"], "k3_maps_frozen": rep["k3_maps_frozen"],
                      "variants": {k: v["n_both_primary"] for k, v in rep["variants"].items()}}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--stim", default=str(HERE / "stimuli_v1") if (HERE / "stimuli_v1").exists()
                   else str(HERE.parent / "stimuli_v1"))
    r.add_argument("--out", required=True)
    r.add_argument("--tasks", default=",".join(TASKS))
    r.add_argument("--sets", default=",".join(LABEL_SETS))
    s = sp.add_parser("select")
    s.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--stim", default=str(HERE.parent / "stimuli_v1"))
    a = ap.parse_args()
    if a.cmd == "check":
        check_reproduction(Path(a.stim), list(TASKS))
    else:
        run(a) if a.cmd == "run" else select(a)


if __name__ == "__main__":
    main()
