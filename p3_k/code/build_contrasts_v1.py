"""v1 (2026-11-25): v0 sets + the generic-magnitude group "mag" (contrast_items_mag_v1.py; external review 2).
The v0 sets are unchanged, so their pairs and splits are identical to contrasts_v0.json.

Build contrasts_v0.json (contrast sentence spec for the appraisal, general-control and non-human-label direction
groups of P3-K; plan v2.2 §3, prereg draft v1 §3) and run the checks.

Inputs : contrast_items_v0.py (authored sentences, this folder)
         ../build_p3k_stimuli.py (EMOTION / REFUSAL lexicons; imported read-only)
         ../stimuli_v0/p3k_pairs.csv (scenario texts), ../stimuli_v0/p3k_readouts_map1.csv (prompt templates)
Outputs: contrasts_v0.json         spec read by `p3k_directions.py contrast --spec contrasts_v0.json`
         contrasts_v0_checks.json  lexicon hits, 6-gram overlap with the task stimuli, duplicates, split counts
         contrasts_v0.sha256       hashes of the spec, the items module and this builder
Deterministic: all randomness from SEED via sha256 (no wall clock). Run: python3 build_contrasts_v0.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)
from contrast_items_v0 import RANDOM_POOL, SETS as SETS_V0  # noqa: E402
from contrast_items_mag_v1 import MAG_SETS  # noqa: E402
SETS = {**SETS_V0, **MAG_SETS}
from build_p3k_stimuli import EMO_RE, REF_RE  # noqa: E402

SEED = 20261215                      # contrast spec seed (splits and random pairings); register with the spec hash
N_RAND_PARTITIONS = 5                # random natural-sentence contrasts: 5 independent random pairings of the pool
N_PAIRS = 40
SLOT = re.compile(r"\[([^\[\]|]*)\|([^\[\]|]*)\]")
# stimuli_v0 sits next to this file on the server (flat layout) and one level up locally
STIM = next(d for d in (os.path.join(HERE, "stimuli_v0"), os.path.join(CODE, "stimuli_v0")) if os.path.isdir(d))

NEUTRAL_PCS_PROCEDURE = {
    "group": "neu_pc",
    "command": "p3k_directions.py neutral-pcs --corpus <LOCAL one-sentence-per-line file> --n 2000 --k 5",
    "steps": [
        "Corpus: a LOCAL neutral English corpus, one sentence per line (candidate: WikiText-103 train; licence and "
        "the exact file to be confirmed and registered). Nothing is downloaded by the pipeline.",
        "Filter: keep lines with 5-40 words, no hit of the EMOTION_WORDS lexicon (build_p3k_stimuli.py), and no "
        "6-gram shared with the task scenarios (stimuli_v0/p3k_pairs.csv); drop exact duplicates.",
        "Sample n = 2000 lines with the registered seed (SEED['subsample'] in p3k_directions.py).",
        "Hidden states: raw text, no chat template, last token, hidden_states[L] (same layer L as all directions).",
        "Directions: mean-center the n x h matrix; the top k = 5 right singular vectors are the directions "
        "neu_pc_1..5 (sign: positive skew of the scores), subspace H_neu_pc = their span.",
        "Use: non-human-label comparison directions (not in the affect set A by default). Report the explained "
        "variance; the top PCs of raw residual activations can be dominated by outlier dimensions, so the whitened "
        "metric of p3k_kill_tests.py applies to them as to every other direction.",
    ],
    "status": "procedure fixed here; corpus file and licence to be registered",
}


def rng_for(*keys) -> random.Random:
    h = hashlib.sha256("|".join(str(k) for k in (SEED,) + keys).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def expand(s: str) -> tuple[str, str]:
    """'[hi|lo]' slots -> (hi sentence, lo sentence); whitespace normalized."""
    assert SLOT.search(s), f"no slot: {s}"
    hi = SLOT.sub(lambda m: m.group(1), s)
    lo = SLOT.sub(lambda m: m.group(2), s)
    norm = lambda x: re.sub(r"\s+", " ", x).strip()
    hi, lo = norm(hi), norm(lo)
    assert hi != lo and "[" not in hi + lo and "|" not in hi + lo, s
    return hi, lo


def split_labels(name: str, n: int) -> list[str]:
    """Seeded half split of the pairs of one set: n // 2 discovery, the rest confirmation."""
    perm = list(range(n))
    rng_for("split", name).shuffle(perm)
    disc = set(perm[: n // 2])
    return ["disc" if i in disc else "conf" for i in range(n)]


def words(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


def ngrams(s: str, n: int = 6) -> set[str]:
    w = words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build() -> dict:
    sets = []
    for name, (group, items, hi_pole, lo_pole) in SETS.items():
        assert len(items) == N_PAIRS, (name, len(items))
        sp = split_labels(name, N_PAIRS)
        pairs = []
        for i, s in enumerate(items):
            hi, lo = expand(s)
            pairs.append({"id": f"{name}_{i + 1:02d}", "hi": hi, "lo": lo, "split": sp[i], "source": s})
        sets.append({"name": name, "group": group, "hi_pole": hi_pole, "lo_pole": lo_pole, "kind": "minimal_pairs",
                     "pairs": pairs})
    assert len(RANDOM_POOL) == 2 * N_PAIRS and len(set(RANDOM_POOL)) == len(RANDOM_POOL)
    for r in range(N_RAND_PARTITIONS):
        name = f"rand_nat_{r}"
        perm = list(range(len(RANDOM_POOL)))
        rng_for("rand_nat", r).shuffle(perm)
        sp = split_labels(name, N_PAIRS)
        pairs = [{"id": f"{name}_{i + 1:02d}", "hi": RANDOM_POOL[perm[2 * i]], "lo": RANDOM_POOL[perm[2 * i + 1]],
                  "split": sp[i]} for i in range(N_PAIRS)]
        sets.append({"name": name, "group": "rand_nat", "hi_pole": "random", "lo_pole": "random",
                     "kind": "random_pairing_of_neutral_pool", "pairs": pairs})
    return {
        "version": "contrasts_v1", "date": "2026-11-25", "seed": SEED,
        "format": ("sets[].pairs[] = {id, hi, lo, split}. Direction (p3k_directions.py contrast, pair mode): "
                   "d = unit(mean over the pairs with split in --splits of (h(hi) - h(lo))), h = last-token "
                   "hidden_states[L] of the raw sentence (no chat template). Group from sets[].group: app -> affect "
                   "set A; ctrl -> control span C; rand_nat -> non-human-label comparison (not in A); mag -> generic-magnitude control (T3, not in A)."),
        "sets": sets,
        "neutral_corpus_pcs": NEUTRAL_PCS_PROCEDURE,
    }


def checks(spec: dict) -> dict:
    scen = []
    with open(os.path.join(STIM, "p3k_pairs.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scen += [r["strong"], r["weak"]]
    scen_ng = set().union(*(ngrams(s) for s in scen))
    tpl = set()
    with open(os.path.join(STIM, "p3k_readouts_map1.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tpl.add(r["prompt_tpl"])
    tpl_ng = set().union(*(ngrams(s) for s in tpl))
    res = {"n_scenario_texts": len(scen), "n_prompt_templates": len(tpl), "per_set": {}}
    seen: dict[str, str] = {}
    dups = []
    set_ng = {}
    for S in spec["sets"]:
        sents = [(p["id"], side, p[side]) for p in S["pairs"] for side in ("hi", "lo")]
        emo = [(i, side, h) for i, side, s in sents for h in EMO_RE.findall(s)]
        ref = [(i, side, h) for i, side, s in sents for h in REF_RE.findall(s)]
        ov_s = sorted({(i, g) for i, _, s in sents for g in ngrams(s) & scen_ng})
        ov_t = sorted({(i, g) for i, _, s in sents for g in ngrams(s) & tpl_ng})
        for i, side, s in sents:
            if S["group"] != "rand_nat":
                if s in seen and seen[s] != i:
                    dups.append((seen[s], i, s))
                seen.setdefault(s, i)
        shared = []
        for p in S["pairs"]:
            a, b = words(p["hi"]), words(p["lo"])
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            m = 0
            while m < min(len(a), len(b)) - k and a[-1 - m] == b[-1 - m]:
                m += 1
            shared.append((k + m) / max(len(a), len(b)))
        set_ng[S["name"]] = set().union(*(ngrams(s) for _, _, s in sents))
        res["per_set"][S["name"]] = {
            "group": S["group"], "n_pairs": len(S["pairs"]),
            "n_disc": sum(p["split"] == "disc" for p in S["pairs"]),
            "n_conf": sum(p["split"] == "conf" for p in S["pairs"]),
            "emotion_lexicon_hits": len(emo), "emotion_hit_examples": emo[:20],
            "refusal_lexicon_hits": len(ref),
            "scenario_6gram_overlap": len(ov_s), "scenario_6gram_examples": ov_s[:10],
            "prompt_template_6gram_overlap": len(ov_t), "prompt_template_6gram_examples": ov_t[:10],
            "minimal_pair_shared_word_frac_median": sorted(shared)[len(shared) // 2] if S["kind"] == "minimal_pairs"
            else None,
            "mean_words_hi": sum(len(words(p["hi"])) for p in S["pairs"]) / len(S["pairs"]),
            "mean_words_lo": sum(len(words(p["lo"])) for p in S["pairs"]) / len(S["pairs"]),
            "mean_abs_word_diff": sum(abs(len(words(p["hi"])) - len(words(p["lo"]))) for p in S["pairs"])
            / len(S["pairs"]),
        }
    names = [S["name"] for S in spec["sets"] if S["group"] != "rand_nat"] + ["rand_nat_0"]
    res["cross_set_shared_6grams"] = {f"{a}~{b}": len(set_ng[a] & set_ng[b])
                                      for i, a in enumerate(names) for b in names[i + 1:]
                                      if set_ng[a] & set_ng[b]}
    res["duplicate_sentences_across_pairs"] = dups
    app = [n for n, v in res["per_set"].items() if v["group"] == "app"]
    res["gate"] = {
        "appraisal_emotion_hits_zero": all(res["per_set"][n]["emotion_lexicon_hits"] == 0 for n in app),
        "scenario_6gram_overlap_zero": all(v["scenario_6gram_overlap"] == 0 for v in res["per_set"].values()),
        "no_duplicates": not dups,
        "politeness_mean_abs_word_diff_le_1": res["per_set"]["ctrl_politeness"]["mean_abs_word_diff"] <= 1.0,
    }
    res["gate"]["pass"] = all(res["gate"].values())
    return res


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    spec = build()
    out = os.path.join(HERE, "contrasts_v1.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    res = checks(spec)
    with open(os.path.join(HERE, "contrasts_v1_checks.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "contrasts_v1.sha256"), "w") as f:
        for n in ("contrasts_v1.json", "contrast_items_v0.py", "contrast_items_mag_v1.py", "build_contrasts_v1.py"):
            f.write(f"{sha(os.path.join(HERE, n))}  {n}\n")
    for n, v in res["per_set"].items():
        print(f"{n:28s} {v['group']:8s} disc/conf {v['n_disc']}/{v['n_conf']}  emo {v['emotion_lexicon_hits']:2d}  "
              f"ref {v['refusal_lexicon_hits']:2d}  scen6 {v['scenario_6gram_overlap']}  "
              f"tpl6 {v['prompt_template_6gram_overlap']}  words hi/lo {v['mean_words_hi']:.1f}/{v['mean_words_lo']:.1f}")
    print("cross-set 6-grams:", res["cross_set_shared_6grams"])
    print("duplicates:", res["duplicate_sentences_across_pairs"])
    print("GATE", res["gate"])


if __name__ == "__main__":
    main()
