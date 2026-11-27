"""Model-free checks for the v3.3 task files (02_設計/06_課題生成規則_v3.3_20261127.md §7). No model is used.

Usage: python3 check_p3k_items_v3.py p3k_items_v3_D1.py [more files ...]
Each file defines TASKS = {task_id: dict(...)}. Checks: required fields, domain / control type against
p3k_v3_task_plan.json, 40 items, exactly one [strong|weak] slot and one {CTRL} placeholder (outside the slot),
slot length <= 15 words per version, duplicates, emotion / refusal lexicon, 6-gram overlap with the 28 development
tasks and between new tasks. Exit code 1 if anything fails.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_p3k_stimuli import EMO_RE, REF_RE, SLOT_RE  # noqa: E402
from p3k_items_n import TASKS_N  # noqa: E402
from p3k_items_v import TASKS_V  # noqa: E402

PLAN = json.load(open(HERE / "p3k_v3_task_plan.json"))
N_ITEMS = 40
FIELDS = ("name", "factor", "domain", "J", "JR", "CAT", "PAIR_Q", "ctrl", "ctrl_role", "items")


def ngrams(s: str, n: int = 6) -> set[str]:
    w = re.findall(r"[a-z']+", s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def versions(text: str) -> tuple[str, str]:
    m = SLOT_RE.search(text)
    s, w = m.group(1), m.group(2)
    return text[:m.start()] + s + text[m.end():], text[:m.start()] + w + text[m.end():]


def load(path: str) -> dict:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TASKS


def main(paths: list[str]) -> int:
    errs = []
    dev_ng = set()
    for T in {**TASKS_V, **TASKS_N}.values():
        for text, _ in T["items"]:
            for v in versions(text):
                dev_ng |= ngrams(v)
    seen_ng: dict[str, str] = {}
    all_tasks = {}
    for p in paths:
        all_tasks.update(load(p))
    for tid, T in all_tasks.items():
        e = lambda msg: errs.append(f"{tid}: {msg}")
        if tid not in PLAN:
            e("task id not in the plan")
            continue
        for f in FIELDS:
            if f not in T:
                e(f"missing field {f}")
        if any(f not in T for f in FIELDS):
            continue
        if T["domain"] != PLAN[tid]["domain"]:
            e(f"domain {T['domain']} != plan {PLAN[tid]['domain']}")
        if T["ctrl"] != PLAN[tid]["ctrl"]:
            e(f"ctrl {T['ctrl']} != plan {PLAN[tid]['ctrl']}")
        for ro in ("J", "JR"):
            if not (len(T[ro]) == 3 and T[ro][2].endswith("?")):
                e(f"{ro} must be (lo, hi, question?)")
        if not (len(T["CAT"]) == 2 and len(T["CAT"][1]) == 4):
            e("CAT must be (question, [4 options])")
        for o in T["CAT"][1]:
            if REF_RE.search(o):
                e(f"refusal word in CAT option: {o}")
        txt = " ".join([T["J"][2], T["JR"][2], T["CAT"][0], T["PAIR_Q"]] + T["CAT"][1] + list(T["J"][:2]) + list(T["JR"][:2]))
        for h in EMO_RE.findall(txt):
            e(f"emotion word in question/options: {h}")
        items = T["items"]
        if len(items) != N_ITEMS:
            e(f"{len(items)} items (need {N_ITEMS})")
        vs = set()
        for i, text in enumerate(items, 1):
            if not isinstance(text, str):
                e(f"item {i} is not a string")
                continue
            if len(SLOT_RE.findall(text)) != 1:
                e(f"item {i}: needs exactly one [strong|weak] slot")
                continue
            if text.count("{CTRL}") != 1:
                e(f"item {i}: needs exactly one {{CTRL}}")
                continue
            m = SLOT_RE.search(text)
            if "{CTRL}" in m.group(0):
                e(f"item {i}: {{CTRL}} inside the slot")
            s, w = m.group(1).strip(), m.group(2).strip()
            if s == w:
                e(f"item {i}: identical versions")
            if max(len(s.split()), len(w.split())) > 15:
                e(f"item {i}: slot longer than 15 words")
            if re.search(r"[{}\[\]]", text.replace("{CTRL}", "").replace(m.group(0), "")):
                e(f"item {i}: stray bracket")
            for v in versions(text):
                key = v.lower().replace("{ctrl}", "")
                if key in vs:
                    e(f"item {i}: duplicate version")
                vs.add(key)
                for h in EMO_RE.findall(v):
                    e(f"item {i}: emotion word '{h}'")
                # 6-grams that contain the placeholder are the mandated control phrase ("reference number {CTRL}",
                # "the village of {CTRL}"), not scenario content; they are excluded from the overlap checks.
                ng = {g for g in ngrams(v.replace("{CTRL}", "X")) if "x" not in g.split()}
                if ng & dev_ng:
                    e(f"item {i}: 6-gram shared with development tasks: {sorted(ng & dev_ng)[:2]}")
                for g in ng:
                    if g in seen_ng and seen_ng[g] != tid:
                        e(f"item {i}: 6-gram shared with {seen_ng[g]}: '{g}'")
                    seen_ng.setdefault(g, tid)
    print(f"{len(all_tasks)} tasks checked, {len(errs)} problems")
    for x in errs[:200]:
        print(" ", x)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
