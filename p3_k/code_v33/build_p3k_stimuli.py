"""Build the P3-K (withdrawal experiment) stimulus set v0 (2026-11-24).

Inputs : p3k_items_v.py, p3k_items_n.py (authored stimuli in this folder).
         Read-only, optional: P2 readout CSVs under ~/git/aiemo-origin-prereg (V1 overlap check only).
Outputs: stimuli_v0/p3k_readouts_map{1..4}.csv  one file per option-label mapping set, identical item_id lists
         stimuli_v0/p3k_pairs.csv               one row per scenario pair (strong / weak / control value)
         stimuli_v0/p3k_label_maps.json         mapping sets and the per-item quad assignment
         stimuli_v0/p3k_checks.json             lexicon / structure / P2-overlap check results
         stimuli_v0/SHA256SUMS                  (written by write_sums(), after README.md exists)

Per task (V1-V16, N1-N12): 30 pairs x 2 versions = 60 scenarios, readouts
  J        1-9 rating (high = more of the judged quantity; the strong version should score higher)
  JR       1-9 rating, reverse-keyed phrasing
  CAT      4 graded options (rank 1 = lowest on J), shown in 4 cyclic rotations (as in P2)
  PAIR     adjacent-rank 2AFC (12, 23, 34), both orders (a/b)
  CTRL     1-9 rating of the control property (lives in the shared text, so strong = weak)
  CTRL_CAT 4 options ordered by the control property, 4 rotations
C1-C4: 30 pairs each (C1/C2 rating, C3/C4 labelled 2AFC in both orders). Not used to estimate F.

Option labels: prompts are rendered from a template with placeholders {L0}..{L3}. Each mapping set is a list of 4
label quads (cyclic shifts of 4 labels = a Latin square, so every label appears at every position equally often).
Per (task, scenario pair) one quad is drawn from a balanced, seeded permutation; both versions of a pair and all
rotations of an item share the quad. PAIR readouts use positions 0 and 1 of the quad.
Deterministic: no wall-clock, all randomness from SEED via sha256.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p3k_items_v import TASKS_V  # noqa: E402
from p3k_items_n import TASKS_N, C_WORDS, C_SENTS, C_ALPHA, C_NUM  # noqa: E402

SEED = 20261124
OUT = os.path.join(HERE, "stimuli_v1")   # v1 = v0 + maps 5 and 6 (external review 2); maps 1-4 are byte-identical
P2_REPO = os.path.expanduser("~/git/aiemo-origin-prereg")

# Candidate single-character labels. Map 1 reuses the P2 letters; maps 3 and 4 share nothing with P2 (prereg K3).
# Single-token status was checked locally on the cached mlx-community Llama-3.1-8B-Instruct tokenizer only
# (verify_label_tokens.py); it MUST be re-verified on the server with the meta-llama tokenizer.
MAP_SETS = {
    1: ["A", "B", "C", "D"],
    2: ["Q", "X", "Z", "J"],
    3: ["ψ", "ξ", "ζ", "θ"],  # psi xi zeta theta
    4: ["§", "¶", "¦", "þ"],  # section, pilcrow, broken bar, thorn
    5: ["Ж", "Щ", "Ф", "Ю"],  # Cyrillic capitals (not used in P2)
    6: ["Д", "Л", "ø", "æ"],  # Cyrillic De, El, o-slash, ash (avoid symbols with a magnitude or rating sense)
}
ROTS = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]  # P2 rotations: rot r shows ranks r+1, r+2, ...
PAIRS = [(0, 1), (1, 2), (2, 3)]

FIELDS = ["item_id", "task", "sev", "scen", "loc", "prompt_qa", "rot", "ranks",
          "ptask", "readout", "kind", "map_set", "labels", "prompt_tpl"]

# ----------------------------------------------------------------------------------------------- lexicons
EMOTION_WORDS = """
angry anger angrily angered furious fury rage raging enraged irritated irritation annoyed annoyance annoying
frustrated frustration frustrating outraged outrage resentful resentment hostile hostility mad
sad sadness sadly sorrow sorrowful grief grieving grieve unhappy miserable misery depressed depression gloomy
melancholy heartbroken despair despairing hopeless lonely loneliness homesick
happy happiness happily joy joyful joyous glad delighted delight cheerful cheer elated pleased pleasure
content contented contentment satisfied satisfaction thrilled excited excitement enthusiastic enthusiasm
amused amusement fun proud pride grateful gratitude thankful relieved relief hope hopeful optimism optimistic
love loving loved adore admire admiration affection fond
fear fearful afraid scared scary frightened fright terrified terror terrifying horror horrified horrifying
panic panicked anxious anxiety nervous nervousness worried worry worrying uneasy dread apprehensive alarmed
disgust disgusted disgusting revulsion repulsed gross contempt hatred hate hated
shame ashamed guilt guilty embarrassed embarrassment humiliated humiliation remorse regret regretful
jealous jealousy envy envious bored boredom surprised surprise surprising astonished amazed amazement shocked
confused confusion curious curiosity awe nostalgia nostalgic sympathy compassion pity caring desire longing
calm calmness relaxed relax peaceful serene tense tension stressed stress distress distressed upset
mood emotion emotional emotions feel feeling feelings felt hurt pain painful suffering suffer cry crying cried
tears smile smiled smiling laugh laughed laughing comfort comfortable uncomfortable pleasant unpleasant
disappointed disappointment disappointing sorry apologetic insecure lonely hysterical
""".split()
REFUSAL_WORDS = """no not none never yes refuse refused decline declined deny denied cannot can't won't sorry
sure okay ok comply compliance accept agree allow unable apologize apologise reject""".split()
EMO_RE = re.compile(r"\b(" + "|".join(sorted(set(map(re.escape, EMOTION_WORDS)), key=len, reverse=True)) + r")\b", re.I)
REF_RE = re.compile(r"\b(" + "|".join(sorted(set(map(re.escape, REFUSAL_WORDS)), key=len, reverse=True)) + r")\b", re.I)
SLOT_RE = re.compile(r"\[([^\[\]|]+)\|([^\[\]|]+)\]")


def rng_for(*keys) -> random.Random:
    h = hashlib.sha256("|".join(str(k) for k in (SEED,) + keys).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def split_slot(text: str):
    m = SLOT_RE.findall(text)
    assert len(m) == 1, f"expected one slot: {text}"
    s, w = m[0]
    full = SLOT_RE.search(text)
    strong = text[:full.start()] + s + text[full.end():]
    weak = text[:full.start()] + w + text[full.end():]
    shared = text[:full.start()] + " " + text[full.end():]
    return strong, weak, shared, s, w


def letters(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


def derive_ctrl(task: str, shared: str, s: str, w: str, ctrl):
    """Control values derived from the shared text; also assert the slot carries none of the control feature."""
    if task == "V5":
        assert not re.search(r"\d", s + w), (task, s, w)
        return len(re.findall(r"\d[\d,.]*", shared))
    if task == "V10":
        assert "`" not in s + w
        return len(re.search(r"`([^`]+)`", shared).group(1))
    if task == "V14":
        return letters(re.match(r"In ([^,]+),", shared).group(1))
    if task == "N1":
        assert "-" not in s + w, (task, s, w)
        return shared.count("-")
    if task == "N2":
        return ord(re.match(r"Solve for ([a-z]):", shared).group(1)) - 96
    if task == "N5":
        return letters(shared.split()[0])
    if task == "N8":
        return letters(re.match(r"At ([^,]+),", shared).group(1))
    if task == "N12":
        return letters(re.match(r"On ([^,]+),", shared).group(1))
    assert ctrl is not None, task
    return ctrl


def rating_prompt(scen: str, lo: str, hi: str, q: str, head: str = "Scenario") -> str:
    return f"{head}: {scen}\nQuestion: On a scale from 1 ({lo}) to 9 ({hi}), {q} Answer with one digit."


def choice_tpl(scen: str, q: str, opts: list[str], head: str = "Scenario") -> str:
    lines = [f"{head}: {scen}", f"Question: {q}"]
    lines += [f"{{L{k}}}. {o}" for k, o in enumerate(opts)]
    lines.append("Answer with one label.")
    return "\n".join(lines)


def render(tpl: str, quad: list[str]) -> str:
    out = tpl
    for k, lab in enumerate(quad):
        out = out.replace(f"{{L{k}}}", lab)
    assert "{L" not in out
    return out


def balanced_assignment(ptask: str, map_set: int, n: int) -> list[int]:
    idx = [i % 4 for i in range(n)]
    rng_for("labels", ptask, map_set).shuffle(idx)
    return idx


def build():
    os.makedirs(OUT, exist_ok=True)
    tasks = {**TASKS_V, **TASKS_N}
    # ---- rows without labels (label placeholders), shared by all map sets
    base_rows = []   # dicts with prompt_tpl; label-dependent rows flagged
    pair_rows = []
    for ptask, T in tasks.items():
        assert len(T["items"]) == 30, (ptask, len(T["items"]))
        assert len(T["CAT"][1]) == 4 and len(T["CTRL_CAT"][1]) == 4
        seen = set()
        for i, (text, ctrl) in enumerate(T["items"], 1):
            strong, weak, shared, s, w = split_slot(text)
            assert s.strip() != w.strip(), text
            for v in (strong, weak):
                assert v not in seen, v
                seen.add(v)
            cval = derive_ctrl(ptask, shared, s, w, ctrl)
            scen = f"{ptask}_{i:02d}"
            pair_rows.append(dict(ptask=ptask, scen=scen, strong=strong, weak=weak, ctrl=cval,
                                  factor=T["factor"], ctrl_name=T["ctrl_name"]))
            for sev, sc in (("strong", strong), ("weak", weak)):
                common = dict(sev=sev, scen=scen, loc=cval, ptask=ptask)
                for ro, kind in (("J", "judgment"), ("JR", "judgment"), ("CTRL", "control")):
                    lo, hi, q = T[ro]
                    p = rating_prompt(sc, lo, hi, q)
                    base_rows.append(dict(common, readout=ro, kind=kind, rot="", ranks="", prompt_tpl=p, labelled=False))
                for ro, kind in (("CAT", "judgment"), ("CTRL_CAT", "control")):
                    q, opts = T[ro]
                    for r, perm in enumerate(ROTS):
                        tpl = choice_tpl(sc, q, [opts[j] for j in perm])
                        base_rows.append(dict(common, readout=ro, kind=kind, rot=str(r),
                                              ranks="".join(str(j + 1) for j in perm), prompt_tpl=tpl, labelled=True))
                q = T["PAIR_Q"]
                opts = T["CAT"][1]
                for a, b in PAIRS:
                    for tag, (x, y) in (("a", (a, b)), ("b", (b, a))):
                        tpl = choice_tpl(sc, q, [opts[x], opts[y]])
                        base_rows.append(dict(common, readout="PAIR", kind="judgment", rot=f"{a + 1}{b + 1}{tag}",
                                              ranks=f"{x + 1}{y + 1}", prompt_tpl=tpl, labelled=True))
    # ---- C1-C4
    for ptask, data in (("C1", C_WORDS), ("C2", C_SENTS), ("C3", C_ALPHA), ("C4", C_NUM)):
        assert len(data) == 30
        for i, (s, w) in enumerate(data, 1):
            scen = f"{ptask}_{i:02d}"
            if ptask == "C1":
                assert 7 <= len(s) <= 9 and 2 <= len(w) <= 4, (s, w)
                pair_rows.append(dict(ptask=ptask, scen=scen, strong=s, weak=w, ctrl="", factor="word length", ctrl_name=""))
                for sev, x in (("strong", s), ("weak", w)):
                    p = rating_prompt(x, "one letter", "nine or more letters", "how many letters does this word have?", head="Word")
                    base_rows.append(dict(sev=sev, scen=scen, loc=len(x), ptask=ptask, readout="J", kind="baseline",
                                          rot="", ranks="", prompt_tpl=p, labelled=False))
            elif ptask == "C2":
                ns, nw = len(s.split()), len(w.split())
                assert 7 <= ns <= 9 and 2 <= nw <= 4, (s, w)
                pair_rows.append(dict(ptask=ptask, scen=scen, strong=s, weak=w, ctrl="", factor="sentence length", ctrl_name=""))
                for sev, x, n in (("strong", s, ns), ("weak", w, nw)):
                    p = rating_prompt(x, "one word", "nine or more words", "how many words does this sentence have?", head="Sentence")
                    base_rows.append(dict(sev=sev, scen=scen, loc=n, ptask=ptask, readout="J", kind="baseline",
                                          rot="", ranks="", prompt_tpl=p, labelled=False))
            else:
                if ptask == "C3":
                    assert s < w, (s, w)
                    q, head, stim = "Which word comes first in alphabetical order?", "Words", f"{s}, {w}"
                else:
                    assert s > w
                    q, head, stim = "Which number is larger?", "Numbers", f"{s}, {w}"
                pair_rows.append(dict(ptask=ptask, scen=scen, strong=str(s), weak=str(w), ctrl="",
                                      factor="alphabetical order" if ptask == "C3" else "numeric magnitude", ctrl_name=""))
                # version a: target (earlier word / larger number) listed first; version b: listed second.
                # ranks: rank of each shown option, 2 = the target.
                for sev, (x, y), rk in (("ord_a", (s, w), "21"), ("ord_b", (w, s), "12")):
                    stim_v = f"{x}, {y}"
                    tpl = choice_tpl(stim_v, q, [str(x), str(y)], head=head)
                    base_rows.append(dict(sev=sev, scen=scen, loc="", ptask=ptask, readout="PAIR", kind="baseline",
                                          rot="", ranks=rk, prompt_tpl=tpl, labelled=True))
    # ---- ids
    for r in base_rows:
        rot = f"_r{r['rot']}" if r["readout"] in ("CAT", "CTRL_CAT") else (f"_{r['rot']}" if r["rot"] else "")
        r["item_id"] = f"P3K_{r['ptask']}_{r['readout']}{rot}_{r['sev']}_{r['scen']}"
        r["task"] = f"{r['ptask']}_{r['readout']}"
    ids = [r["item_id"] for r in base_rows]
    assert len(ids) == len(set(ids)), "duplicate item ids"

    # ---- label assignment and per-map files
    assign = {}
    all_ptasks = list(tasks) + ["C1", "C2", "C3", "C4"]
    for m, labs in MAP_SETS.items():
        quads = [[labs[(k + s) % 4] for k in range(4)] for s in range(4)]
        assign[m] = {"labels": labs, "quads": quads, "per_item": {}}
        for pt in all_ptasks:
            idx = balanced_assignment(pt, m, 30)
            for i, q in enumerate(idx, 1):
                assign[m]["per_item"][f"{pt}_{i:02d}"] = q
        rows = []
        for r in base_rows:
            out = {k: r[k] for k in FIELDS if k in r}
            out["map_set"] = m
            if r["labelled"]:
                quad = quads[assign[m]["per_item"][r["scen"]]]
                n_opt = 4 if r["readout"] in ("CAT", "CTRL_CAT") else 2
                out["labels"] = "|".join(quad[:n_opt])
                out["prompt_qa"] = render(r["prompt_tpl"], quad)
            else:
                out["labels"] = ""
                out["prompt_qa"] = r["prompt_tpl"]
            rows.append(out)
        with open(os.path.join(OUT, f"p3k_readouts_map{m}.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
    with open(os.path.join(OUT, "p3k_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["ptask", "scen", "strong", "weak", "ctrl", "factor", "ctrl_name"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(pair_rows)
    with open(os.path.join(OUT, "p3k_label_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "maps": assign}, f, ensure_ascii=False, indent=1, sort_keys=True)
    return tasks, base_rows, pair_rows, assign


def checks(tasks, base_rows, pair_rows, assign):
    res = {}
    # 1. emotion lexicon over every rendered prompt (scenario, question, options) in every map set
    emo_hits = []
    for m in MAP_SETS:
        with open(os.path.join(OUT, f"p3k_readouts_map{m}.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for h in EMO_RE.findall(r["prompt_qa"]):
                    emo_hits.append((m, r["item_id"], h))
    res["emotion_lexicon_size"] = len(set(EMOTION_WORDS))
    res["emotion_hits"] = len(emo_hits)
    res["emotion_hit_examples"] = emo_hits[:20]
    # 2. refusal/compliance lexicon over answer options and labels
    ref_hits = []
    for pt, T in tasks.items():
        for o in T["CAT"][1] + T["CTRL_CAT"][1]:
            ref_hits += [(pt, o, h) for h in REF_RE.findall(o)]
    for m, labs in MAP_SETS.items():
        ref_hits += [(f"map{m}", l, l) for l in labs if REF_RE.search(l)]
    res["refusal_lexicon_size"] = len(set(REFUSAL_WORDS))
    res["refusal_hits_in_answers"] = len(ref_hits)
    res["refusal_hit_examples"] = ref_hits[:20]
    # 3. structure counts
    cnt = {}
    for r in base_rows:
        cnt.setdefault(r["ptask"], {}).setdefault(r["readout"], 0)
        cnt[r["ptask"]][r["readout"]] += 1
    res["rows_per_task_readout"] = cnt
    res["rows_total_per_map_file"] = len(base_rows)
    res["pairs_per_task"] = {pt: sum(1 for p in pair_rows if p["ptask"] == pt) for pt in cnt}
    # 4. label counterbalance: quad usage per task and map
    bal = {}
    for m in MAP_SETS:
        for pt in cnt:
            c = [0, 0, 0, 0]
            for i in range(1, 31):
                c[assign[m]["per_item"][f"{pt}_{i:02d}"]] += 1
            assert max(c) - min(c) <= 1
            bal.setdefault(str(m), {})[pt] = c
    res["quad_usage_per_task"] = bal
    # 5. V1 overlap with P2 scenario text (read-only scan of P2 readout CSVs, if present)
    p2_scen = set()
    for fn in sorted(glob.glob(os.path.join(P2_REPO, "**", "readout_R9*.csv"), recursive=True)):
        with open(fn, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = r.get("prompt_qa", "")
                if t.startswith("Scenario: "):
                    first = t[len("Scenario: "):].split("\n")[0]
                    for sent in re.split(r"(?<=\.)\s+", first):
                        if len(sent) > 20:
                            p2_scen.add(sent.strip().lower())
    v1 = [p for p in pair_rows if p["ptask"] == "V1"]

    def ngrams(s, n=6):
        w = re.findall(r"[a-z']+", s.lower())
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    p2_ng = set().union(*(ngrams(s) for s in p2_scen)) if p2_scen else set()
    exact = [p["scen"] for p in v1 for v in (p["strong"], p["weak"]) if v.strip().lower() in p2_scen]
    ng = sorted({(p["scen"], g) for p in v1 for v in (p["strong"], p["weak"]) for g in ngrams(v) & p2_ng})
    res["v1_vs_p2"] = dict(p2_sentences_scanned=len(p2_scen), exact_sentence_matches=exact, shared_6grams=ng)
    # 6. control values: strong and weak share the control text by construction; report spread per task
    spread = {}
    for pt in tasks:
        vals = [p["ctrl"] for p in pair_rows if p["ptask"] == pt]
        spread[pt] = {"min": min(vals), "max": max(vals), "n_distinct": len(set(vals))}
    res["ctrl_spread"] = spread
    with open(os.path.join(OUT, "p3k_checks.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    return res


def write_sums():
    names = sorted(n for n in os.listdir(OUT) if n != "SHA256SUMS" and not n.startswith("."))
    lines = []
    for n in names:
        with open(os.path.join(OUT, n), "rb") as f:
            lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {n}")
    for n in ("build_p3k_stimuli.py", "p3k_items_v.py", "p3k_items_n.py", "verify_label_tokens.py"):
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            with open(p, "rb") as f:
                lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  ../{n}")
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    tasks, base_rows, pair_rows, assign = build()
    res = checks(tasks, base_rows, pair_rows, assign)
    write_sums()
    print("rows per map file:", res["rows_total_per_map_file"])
    print("emotion hits:", res["emotion_hits"], res["emotion_hit_examples"][:10])
    print("refusal hits in answers:", res["refusal_hits_in_answers"], res["refusal_hit_examples"])
    print("V1 vs P2:", res["v1_vs_p2"]["p2_sentences_scanned"], res["v1_vs_p2"]["exact_sentence_matches"],
          res["v1_vs_p2"]["shared_6grams"][:10])
    for pt, s in res["ctrl_spread"].items():
        print(pt, s)
