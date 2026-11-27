"""Build the P3-K v3.3 stimulus set (confirmatory cohort; rule: 02_設計/06_課題生成規則_v3.3_20261127.md).

Inputs : p3k_items_v3_D1.py .. p3k_items_v3_D6.py (48 authored tasks, 40 scenarios each), p3k_v3_task_plan.json.
Outputs: stimuli_v3/p3k_pilot.csv             eligibility test rows: J and CTRL only, the 10 pilot scenarios per task
         stimuli_v3/p3k_readouts_map{M}.csv   main rows (30 scenarios per task), one file per label set
         stimuli_v3/p3k_pairs.csv             one row per scenario (pilot and main), with the control value
         stimuli_v3/p3k_label_maps.json       label sets, quads and per-scenario quad assignment
         stimuli_v3/p3k_checks.json           lexicon / structure / balance checks
         stimuli_v3/SHA256SUMS

Differences from stimuli_v1 (all fixed by the rule and by stage D, prereg v3.3):
  * control values are generated, not authored: NAME = an invented name whose letter count falls in one of 4 bins
    (3-4, 5-6, 7-9, 10-12), CODE = a digit string with 2-3, 4-5, 6-7 or 8-9 digits; 10 scenarios per bin, seeded
    (SEED_CTRL), shared by the strong and weak version, independent of the scenario content.
  * 10 of the 40 scenarios per task are pilot scenarios (seeded, SEED_SPLIT); the 30 main scenarios are renumbered
    01-30 in their original order, pilot scenarios P01-P10.
  * PAIR uses the rank pairs 13, 24, 14 in both orders (stage D variant D2). CAT and CTRL_CAT keep the 4 cyclic
    rotations.
  * label sets: 1 = A B C D, 2 = Q X Z J (primary task entry), 3 = psi xi zeta theta, 7 = H K R W, 8 = G N P T
    (K3 candidates; 7 and 8 are 'Ka' and 'Kb' of stage D).
Deterministic: no wall-clock; all randomness from the seeds via sha256.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_p3k_stimuli as B  # noqa: E402  (rating_prompt, choice_tpl, render, split_slot, EMO_RE, REF_RE)

SEED_CTRL, SEED_SPLIT, SEED_LABELS, SEED_NAMES = 20261227, 20261228, 20261229, 20261230
OUT = os.path.join(HERE, "stimuli_v3")
DOMAIN_FILES = [os.path.join(HERE, f"p3k_items_v3_D{d}.py") for d in range(1, 7)]
MAP_SETS = {"1": ["A", "B", "C", "D"], "2": ["Q", "X", "Z", "J"], "3": ["ψ", "ξ", "ζ", "θ"],
            "7": ["H", "K", "R", "W"], "8": ["G", "N", "P", "T"]}
ROTS = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
PAIRS = [(0, 2), (1, 3), (0, 3)]                       # D2
N_SCEN, N_PILOT = 40, 10
NAME_BINS = [(3, 4), (5, 6), (7, 9), (10, 12)]
CODE_BINS = [(2, 3), (4, 5), (6, 7), (8, 9)]
CTRL_Q = {
    "NAME": dict(CTRL=("a very short name", "a very long name", "how many letters are in the name of {role}?"),
                 CTRL_CAT=("How many letters are in the name of {role}?",
                           ["three or four letters", "five or six letters", "seven to nine letters",
                            "ten or more letters"])),
    "CODE": dict(CTRL=("very few digits", "very many digits", "how many digits does the reference number have?"),
                 CTRL_CAT=("How many digits does the reference number have?",
                           ["two or three digits", "four or five digits", "six or seven digits",
                            "eight or nine digits"])),
}
FIELDS = ["item_id", "task", "sev", "scen", "loc", "prompt_qa", "rot", "ranks",
          "ptask", "readout", "kind", "map_set", "labels", "prompt_tpl"]
CONS, VOWS = "bdfgklmnprstvz", "aeiou"


def rng_for(*keys) -> random.Random:
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def load_tasks() -> dict:
    tasks = {}
    for p in DOMAIN_FILES:
        spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tasks.update(mod.TASKS)
    plan = json.load(open(os.path.join(HERE, "p3k_v3_task_plan.json")))
    assert set(tasks) == set(plan), sorted(set(tasks) ^ set(plan))
    for t, T in tasks.items():
        assert T["ctrl"] == plan[t]["ctrl"] and T["domain"] == plan[t]["domain"], t
        assert len(T["items"]) == N_SCEN, (t, len(T["items"]))
    order = sorted(tasks, key=lambda t: (t[0] != "V", int(t[1:])))
    return {t: tasks[t] for t in order}


def english_words() -> set[str]:
    p = "/usr/share/dict/words"
    return {w.strip().lower() for w in open(p)} if os.path.exists(p) else set()


def make_name(r: random.Random, n: int, banned: set[str]) -> str:
    while True:
        s = "".join(r.choice(CONS) if i % 2 == 0 else r.choice(VOWS) for i in range(n))
        if s not in banned and not B.EMO_RE.search(s) and not B.REF_RE.search(s):
            banned.add(s)
            return s.capitalize()


def ctrl_values(task: str, kind: str, banned: set[str]) -> list[tuple[str, int]]:
    """(rendered value, control value) per scenario; 10 scenarios per bin, seeded."""
    bins = NAME_BINS if kind == "NAME" else CODE_BINS
    idx = [i % 4 for i in range(N_SCEN)]
    rng_for(SEED_CTRL, "bins", task).shuffle(idx)
    r = rng_for(SEED_NAMES, "values", task)
    out = []
    for b in idx:
        n = r.randint(*bins[b])
        if kind == "NAME":
            out.append((make_name(r, n, banned), n))
        else:
            out.append((str(r.randint(1, 9)) + "".join(str(r.randint(0, 9)) for _ in range(n - 1)), n))
    return out


def build():
    os.makedirs(OUT, exist_ok=True)
    tasks = load_tasks()
    banned = english_words()
    pair_rows, pilot_rows, main_rows = [], [], []
    for t, T in tasks.items():
        cv = ctrl_values(t, T["ctrl"], banned)
        pil = sorted(rng_for(SEED_SPLIT, "pilot", t).sample(range(N_SCEN), N_PILOT))
        main = [i for i in range(N_SCEN) if i not in pil]
        sid = {i: f"{t}_P{k + 1:02d}" for k, i in enumerate(pil)} | {i: f"{t}_{k + 1:02d}" for k, i in enumerate(main)}
        cq = {ro: tuple(x.replace("{role}", T["ctrl_role"]) if isinstance(x, str) else x for x in q)
              for ro, q in CTRL_Q[T["ctrl"]].items()}
        for i, text in enumerate(T["items"]):
            val, n = cv[i]
            strong, weak, _shared, s, w = B.split_slot(text.replace("{CTRL}", val))
            scen = sid[i]
            pilot = i in pil
            pair_rows.append(dict(ptask=t, scen=scen, orig=i + 1, pilot=int(pilot), strong=strong, weak=weak,
                                  ctrl=n, ctrl_value=val, ctrl_kind=T["ctrl"], domain=T["domain"],
                                  factor=T["factor"]))
            for sev, sc in (("strong", strong), ("weak", weak)):
                common = dict(sev=sev, scen=scen, loc=n, ptask=t)
                rating = {"J": T["J"], "JR": T["JR"], "CTRL": cq["CTRL"]}
                for ro, kind in (("J", "judgment"), ("JR", "judgment"), ("CTRL", "control")):
                    lo, hi, q = rating[ro]
                    row = dict(common, readout=ro, kind=kind, rot="", ranks="",
                               prompt_tpl=B.rating_prompt(sc, lo, hi, q), labelled=False)
                    if pilot:
                        if ro in ("J", "CTRL"):
                            pilot_rows.append(row)
                    else:
                        main_rows.append(row)
                if pilot:
                    continue
                for ro, kind, (q, opts) in (("CAT", "judgment", T["CAT"]), ("CTRL_CAT", "control", cq["CTRL_CAT"])):
                    for r_, perm in enumerate(ROTS):
                        main_rows.append(dict(common, readout=ro, kind=kind, rot=str(r_),
                                              ranks="".join(str(j + 1) for j in perm), labelled=True,
                                              prompt_tpl=B.choice_tpl(sc, q, [opts[j] for j in perm])))
                opts = T["CAT"][1]
                for a, b in PAIRS:
                    for tag, (x, y) in (("a", (a, b)), ("b", (b, a))):
                        main_rows.append(dict(common, readout="PAIR", kind="judgment", rot=f"{a + 1}{b + 1}{tag}",
                                              ranks=f"{x + 1}{y + 1}", labelled=True,
                                              prompt_tpl=B.choice_tpl(sc, T["PAIR_Q"], [opts[x], opts[y]])))
    for r in pilot_rows + main_rows:
        rot = f"_r{r['rot']}" if r["readout"] in ("CAT", "CTRL_CAT") else (f"_{r['rot']}" if r["rot"] else "")
        r["item_id"] = f"P3K3_{r['ptask']}_{r['readout']}{rot}_{r['sev']}_{r['scen']}"
        r["task"] = f"{r['ptask']}_{r['readout']}"
    ids = [r["item_id"] for r in pilot_rows + main_rows]
    assert len(ids) == len(set(ids)), "duplicate item ids"

    def write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)

    write(os.path.join(OUT, "p3k_pilot.csv"),
          [{k: r[k] for k in FIELDS if k in r} | {"map_set": "", "labels": "", "prompt_qa": r["prompt_tpl"]}
           for r in pilot_rows])
    assign = {}
    for m, labs in MAP_SETS.items():
        quads = [[labs[(k + s) % 4] for k in range(4)] for s in range(4)]
        assign[m] = {"labels": labs, "quads": quads, "per_item": {}}
        for t in tasks:
            idx = [i % 4 for i in range(N_SCEN - N_PILOT)]
            rng_for(SEED_LABELS, "labels", t, m).shuffle(idx)
            for k, q in enumerate(idx, 1):
                assign[m]["per_item"][f"{t}_{k:02d}"] = q
        rows = []
        for r in main_rows:
            o = {k: r[k] for k in FIELDS if k in r}
            o["map_set"] = m
            if r["labelled"]:
                quad = quads[assign[m]["per_item"][r["scen"]]]
                o["labels"] = "|".join(quad[:4 if r["readout"] in ("CAT", "CTRL_CAT") else 2])
                o["prompt_qa"] = B.render(r["prompt_tpl"], quad)
            else:
                o["labels"], o["prompt_qa"] = "", r["prompt_tpl"]
            rows.append(o)
        write(os.path.join(OUT, f"p3k_readouts_map{m}.csv"), rows)
    with open(os.path.join(OUT, "p3k_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(pair_rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(pair_rows)
    json.dump({"seeds": {"ctrl": SEED_CTRL, "split": SEED_SPLIT, "labels": SEED_LABELS, "names": SEED_NAMES},
               "maps": assign}, open(os.path.join(OUT, "p3k_label_maps.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)
    return tasks, pair_rows, pilot_rows, main_rows


def checks(tasks, pair_rows, pilot_rows, main_rows):
    res = {"n_tasks": len(tasks), "n_pilot_rows": len(pilot_rows), "n_main_rows_per_map": len(main_rows)}
    hits = []
    for m in MAP_SETS:
        for r in csv.DictReader(open(os.path.join(OUT, f"p3k_readouts_map{m}.csv"), encoding="utf-8")):
            hits += [(m, r["item_id"], h) for h in B.EMO_RE.findall(r["prompt_qa"])]
    for r in pilot_rows:
        hits += [("pilot", r["item_id"], h) for h in B.EMO_RE.findall(r["prompt_tpl"])]
    res["emotion_hits"], res["emotion_hit_examples"] = len(hits), hits[:20]
    ref = [(t, o) for t, T in tasks.items() for o in T["CAT"][1] if B.REF_RE.search(o)]
    res["refusal_hits_in_options"] = ref
    cnt = {}
    for r in main_rows:
        cnt.setdefault(r["ptask"], {}).setdefault(r["readout"], 0)
        cnt[r["ptask"]][r["readout"]] += 1
    res["main_rows_per_task_readout"] = cnt
    res["ctrl_bins_per_task"] = {}
    for t, T in tasks.items():
        bins = NAME_BINS if T["ctrl"] == "NAME" else CODE_BINS
        vals = [p["ctrl"] for p in pair_rows if p["ptask"] == t]
        c = [sum(lo <= v <= hi for v in vals) for lo, hi in bins]
        assert c == [10, 10, 10, 10], (t, c)
        res["ctrl_bins_per_task"][t] = c
    res["domains"] = {d: {"V": [t for t, T in tasks.items() if T["domain"] == d and t[0] == "V"],
                          "N": [t for t, T in tasks.items() if T["domain"] == d and t[0] == "N"]}
                      for d in sorted({T["domain"] for T in tasks.values()})}
    json.dump(res, open(os.path.join(OUT, "p3k_checks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def write_sums():
    names = sorted(n for n in os.listdir(OUT) if n != "SHA256SUMS" and not n.startswith("."))
    lines = [f"{hashlib.sha256(open(os.path.join(OUT, n), 'rb').read()).hexdigest()}  {n}" for n in names]
    for n in ["build_p3k_stimuli_v3.py", "build_p3k_stimuli.py", "p3k_v3_task_plan.json", "check_p3k_items_v3.py"] + \
            [os.path.basename(p) for p in DOMAIN_FILES]:
        lines.append(f"{hashlib.sha256(open(os.path.join(HERE, n), 'rb').read()).hexdigest()}  ../{n}")
    open(os.path.join(OUT, "SHA256SUMS"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    tasks, pair_rows, pilot_rows, main_rows = build()
    res = checks(tasks, pair_rows, pilot_rows, main_rows)
    write_sums()
    print({k: res[k] for k in ("n_tasks", "n_pilot_rows", "n_main_rows_per_map", "emotion_hits",
                               "refusal_hits_in_options")})
    print(res["emotion_hit_examples"][:10])
