"""P3-K step 2: direction sets and candidate subspaces in the layer-L residual stream (plan v2.2 §3, prereg draft §3).

Subcommands (all write an .npz with D (n x h, unit rows), names, groups, and candidate subspaces H_* (h x k)):
  contrast      contrast directions. (1) closing-sentence sets, P2 procedure (body.tex §4.1, s1o_rating.py):
                w_x = unit(mean_h(context + closing in x) - mean_h(context + neutral closing)), v = unit(w_hi - w_lo).
                (2) minimal-pair spec contrasts_v0.json (build_contrasts_v0.py): appraisal (goal obstruction,
                agency other/self, certainty), general control (refusal, confidence, politeness, length, topic) and
                random natural-sentence contrasts; d = unit(mean(h(hi) - h(lo))) over the --splits pairs.
  neutral-pcs   non-human-label directions: top-k PCs of last-token states on a LOCAL neutral corpus.
  vad-ext       H_VA^ext / H_VAD^ext: ridge from last-token hidden states to independent human V/A(/D) ratings.
                Loader expects a LOCAL file (NRC-VAD cannot be redistributed; EmoBank CC-BY-SA). Nothing is downloaded.
  cat           category directions (mean of a category minus mean of the neutral texts) from a LOCAL labeled CSV
                (GoEmotions licence still to be confirmed: 10_人間評定データ入手性), and H_cat = top-k PCs.
  sun-va        H_VA^Sun: Sun et al. (arXiv 2604.03147) re-implementation (GoEmotions steering vectors, three
                verbatim self-report templates, PCA + ridge, LOLO-CV choice of k and lambda, Gram-Schmidt) with the
                label-level reproduction check and an NRC-VAD-supervised variant. `sun-va --selftest` runs on the CPU.
  random        variance-matched random directions (P2 fast_var_matched, copied in p3k_common) for every target.
  assemble      merge parts into one directions file for p3k_kill_tests.py.
  select-layer  layer choice without task gradients (prereg §2): split-half reproducibility of category directions.
GPU subcommands: contrast, vad-ext, cat, select-layer, sun-va, neutral-pcs. CPU: random, assemble, sun-va --selftest.

Groups used downstream (p3k_kill_tests.py --affect_groups, default emo,cat,app,va_ext,va_sun): A = span of these rows.
  emo (emotion-intensity contrasts), app (appraisal contrasts), cat, va_ext, va_sun, ctrl (general control; span C),
  rand (variance-matched random; evaluation only).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from p3k_common import fast_var_matched, orth_basis, unit

SEED = {"rand": 20261211, "ridge_cv": 20261212, "layer_split": 20261213, "subsample": 20261214}


# ------------------------------------------------------------------ model access (GPU subcommands only)
def load(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    return tok, model


def hidden(tok, model, texts: list[str], L: int) -> np.ndarray:
    from p3k_gradients import last_hidden_batched
    return last_hidden_batched(model, tok, texts, L)


def save_part(path: str, D, names, groups, meta: dict, **subspaces):
    np.savez(path, D=np.asarray(D, float), names=np.array(names), groups=np.array(groups),
             meta=json.dumps(meta), **subspaces)
    print(json.dumps({"out": path, "n": len(names), "subspaces": {k: v.shape for k, v in subspaces.items()}},
                     default=str))


# ------------------------------------------------------------------ contrast (P2 procedure; minimal-pair mode)
def pair_contrast_dirs(H_hi: np.ndarray, H_lo: np.ndarray) -> np.ndarray:
    """Minimal-pair mode: d = unit(mean_i (h(hi_i) - h(lo_i))). [OURS] no neutral baseline: the pairs are minimal pairs,
    so the shared text cancels within each pair (P2's neutral closing has no counterpart here)."""
    return unit((H_hi - H_lo).mean(0))


def cmd_contrast(a):
    """Two spec formats.
    (1) P2 closing-sentence format:
    {"contexts": [{"text": "...", "split": "disc" | "conf"}, ...]   (optional; without contexts the closings stand alone)
     "neutral": ["closing", ...],
     "contrasts": {"<name>": {"group": "emo" | "app" | "ctrl", "hi": ["closing", ...], "lo": ["closing", ...]}, ...}}
    Closing k is attached to context i with k = i mod K (P2: ks = i % K); hi, lo and neutral must have the same K.
    Only contexts with split in --splits are used (A is built from the discovery items, plan §1).
    (2) Minimal-pair format (contrasts_v0.json, build_contrasts_v0.py):
    {"sets": [{"name", "group", "pairs": [{"id", "hi", "lo", "split"}, ...]}, ...]}
    One direction per set from the pairs with split in --splits (pair_contrast_dirs). --sets keeps only the listed
    set names or groups (comma-separated; empty = all)."""
    spec = json.load(open(a.spec, encoding="utf-8"))
    tok, model = load(a.model, a.revision)
    splits = a.splits.split(",")
    if "sets" in spec:
        keep = set(a.sets.split(",")) if a.sets else None
        D, names, groups, npairs = [], [], [], {}
        for S in spec["sets"]:
            if keep and S["name"] not in keep and S["group"] not in keep:
                continue
            P = [p for p in S["pairs"] if p["split"] in splits]
            H_hi = hidden(tok, model, [p["hi"] for p in P], a.layer).astype(np.float64)
            H_lo = hidden(tok, model, [p["lo"] for p in P], a.layer).astype(np.float64)
            D.append(pair_contrast_dirs(H_hi, H_lo))
            names.append(S["name"])
            groups.append(S["group"])
            npairs[S["name"]] = len(P)
        app = [d for d, g in zip(D, groups) if g == "app"]
        subs = {"H_app": orth_basis(np.stack(app, 1))} if app else {}   # plan §3: H_app = span of the appraisal dirs
        save_part(a.out, D, names, groups, {"cmd": "contrast", "mode": "minimal_pairs", "spec": a.spec,
                                            "spec_version": spec.get("version"), "layer": a.layer,
                                            "splits": splits, "n_pairs": npairs}, **subs)
        return
    ctx = [c["text"] for c in spec.get("contexts", []) if c.get("split", "disc") in splits]
    K = len(spec["neutral"])

    def texts(closings):
        assert len(closings) == K, "every closing set must have the neutral set's length (P2 matching)"
        if not ctx:
            return list(closings)
        return [f"{c} {closings[i % K]}" for i, c in enumerate(ctx)]

    m_neu = hidden(tok, model, texts(spec["neutral"]), a.layer).mean(0)
    D, names, groups = [], [], []
    for nm, c in spec["contrasts"].items():
        w_hi = unit(hidden(tok, model, texts(c["hi"]), a.layer).mean(0) - m_neu)
        w_lo = unit(hidden(tok, model, texts(c["lo"]), a.layer).mean(0) - m_neu)
        D.append(unit(w_hi - w_lo))
        names.append(nm)
        groups.append(c["group"])
    save_part(a.out, D, names, groups, {"cmd": "contrast", "spec": a.spec, "layer": a.layer, "n_ctx": len(ctx)})


# ------------------------------------------------------------------ neutral-corpus PCs (non-human-label directions)
def neutral_pcs(H: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k right singular vectors of the mean-centered hidden states; sign fixed so that the scores have positive
    skew. Returns (k x h directions, explained-variance fractions)."""
    Hc = H - H.mean(0)
    _, s, Wt = np.linalg.svd(Hc, full_matrices=False)
    D = Wt[:k].copy()
    sc = Hc @ D.T
    D *= np.where(((sc - sc.mean(0)) ** 3).mean(0) < 0, -1.0, 1.0)[:, None]
    return D, (s[:k] ** 2) / np.sum(s ** 2)


def cmd_neutral_pcs(a):
    """Procedure: contrasts_v0.json['neutral_corpus_pcs']. LOCAL corpus, one sentence per line; filter (5-40 words, no
    emotion-lexicon hit, no 6-gram shared with the task scenarios, no duplicates); seeded sample of --n lines."""
    import re
    import sys
    for d in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        sys.path.insert(0, str(d))
    from build_p3k_stimuli import EMO_RE
    toks = lambda s: re.findall(r"[a-z']+", s.lower())
    ng = lambda s: {" ".join(toks(s)[i:i + 6]) for i in range(len(toks(s)) - 5)}
    scen = set()
    for r in csv.DictReader(open(a.stim_pairs, encoding="utf-8")):
        scen |= ng(r["strong"]) | ng(r["weak"])
    seen, lines = set(), []
    for ln in open(a.corpus, encoding="utf-8"):
        s = ln.strip()
        if s and s not in seen and a.min_words <= len(toks(s)) <= a.max_words and not EMO_RE.search(s) \
                and not (ng(s) & scen):
            seen.add(s)
            lines.append(s)
    assert len(lines) >= a.n, f"only {len(lines)} lines pass the filter"
    idx = np.sort(np.random.default_rng(SEED["subsample"]).choice(len(lines), a.n, replace=False))
    tok, model = load(a.model, a.revision)
    H = hidden(tok, model, [lines[i] for i in idx], a.layer).astype(np.float64)
    D, ev = neutral_pcs(H, a.k)
    save_part(a.out, D, [f"neu_pc_{i + 1}" for i in range(a.k)], ["neu_pc"] * a.k,
              {"cmd": "neutral-pcs", "corpus": a.corpus, "n_pass_filter": len(lines), "n": a.n, "layer": a.layer,
               "explained_var": ev.tolist()}, H_neu_pc=orth_basis(D.T))


# ------------------------------------------------------------------ external human V/A(/D) ratings
def read_ratings(path: str, fmt: str, dims: list[str], text_col: str | None, cols: dict | None):
    """Return (texts, Y). fmt:
      nrc      NRC VAD Lexicon (local copy). Tab-separated; a header row with 'term' (or 'Word') and valence/arousal/
               dominance columns is detected; without a header the order term, V, A, D is assumed.
               TODO(server): confirm the column names and value range of the v2.1 file before use.
      emobank  EmoBank emobank.csv (columns id, split, V, A, D, text).
      csv      any CSV; --text_col and --cols V=colname,A=colname[,D=colname]."""
    if fmt == "emobank":
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        return [r["text"] for r in rows], np.array([[float(r[d]) for d in dims] for r in rows])
    if fmt == "csv":
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        return [r[text_col] for r in rows], np.array([[float(r[cols[d]]) for d in dims] for r in rows])
    if fmt == "nrc":
        lines = [ln.rstrip("\n").split("\t") for ln in open(path, encoding="utf-8") if ln.strip()]
        head = [h.strip().lower() for h in lines[0]]
        if any(h in ("term", "word") for h in head):
            ti = head.index("term") if "term" in head else head.index("word")
            name = {"V": "valence", "A": "arousal", "D": "dominance"}
            ci = [head.index(name[d]) for d in dims]
            body = lines[1:]
        else:
            ti, ci, body = 0, [{"V": 1, "A": 2, "D": 3}[d] for d in dims], lines
        return [r[ti] for r in body], np.array([[float(r[c]) for c in ci] for r in body])
    raise ValueError(fmt)


def ridge_cv(X: np.ndarray, y: np.ndarray, lams: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, float, float]:
    """Ridge on centered X with the penalty chosen by k-fold CV (Pearson r of held-out predictions).
    Returns (coefficient vector, chosen lambda, CV r)."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % k
    pred = np.zeros((len(lams), len(y)))
    for f in range(k):
        tr, te = fold != f, fold == f
        mu, my = X[tr].mean(0), y[tr].mean()
        U, s, Vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
        uy = U.T @ (y[tr] - my)
        for i, lam in enumerate(lams):
            pred[i, te] = (X[te] - mu) @ (Vt.T @ (s / (s ** 2 + lam) * uy)) + my
    r = np.array([np.corrcoef(p, y)[0, 1] for p in pred])
    best = int(np.argmax(r))
    mu, my = X.mean(0), y.mean()
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    coef = Vt.T @ (s / (s ** 2 + lams[best]) * (U.T @ (y - my)))
    return coef, float(lams[best]), float(r[best])


def cmd_vad_ext(a):
    dims = a.dims.split(",")
    cols = dict(kv.split("=") for kv in a.cols.split(",")) if a.cols else None
    texts, Y = read_ratings(a.data, a.format, dims, a.text_col, cols)
    if a.max_items and len(texts) > a.max_items:
        idx = np.sort(np.random.default_rng(SEED["subsample"]).choice(len(texts), a.max_items, replace=False))
        texts, Y = [texts[i] for i in idx], Y[idx]
    texts = [a.template.format(text=t) for t in texts]
    tok, model = load(a.model, a.revision)
    X = hidden(tok, model, texts, a.layer).astype(np.float64)
    sc = np.linalg.svd(X - X.mean(0), compute_uv=False)
    lams = (sc[0] ** 2) * np.logspace(-6, 1, 15)
    D, info = [], {}
    for j, d in enumerate(dims):
        coef, lam, r = ridge_cv(X, Y[:, j], lams, 5, SEED["ridge_cv"])
        # coefficient (decoding) direction; the encoding (pattern) direction cov @ coef is saved for sensitivity
        pat = (X - X.mean(0)).T @ ((X - X.mean(0)) @ coef) / (len(X) - 1)
        D.append(unit(coef))
        info[d] = {"lambda": lam, "cv_r": r, "cos_coef_pattern": float(unit(coef) @ unit(pat))}
        info[d]["pattern"] = unit(pat).tolist() if a.save_pattern else None
    tagH = {2: "H_VA_ext", 3: "H_VAD_ext"}[len(dims)] + ("" if a.name == "ext" else f"_{a.name}")  # e.g. H_VA_ext_emobank
    save_part(a.out, D, [f"{a.name}_{d}" for d in dims], ["va_ext"] * len(dims),
              {"cmd": "vad-ext", "data": a.data, "format": a.format, "n": len(texts), "layer": a.layer,
               "template": a.template, "fit": {k: {kk: vv for kk, vv in v.items() if kk != "pattern"}
                                               for k, v in info.items()}},
              **{tagH: orth_basis(np.stack(D, 1))})


# ------------------------------------------------------------------ human category directions
def read_labeled(path: str, text_col: str, label_col: str, sep: str) -> tuple[list[str], list[list[str]]]:
    """Local CSV/TSV; label_col may hold several labels separated by sep (GoEmotions style). TODO(server): licence."""
    delim = "\t" if path.endswith(".tsv") else ","
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter=delim))
    return [r[text_col] for r in rows], [r[label_col].split(sep) for r in rows]


def category_dirs(H: np.ndarray, labels: list[list[str]], neutral: str, min_n: int, single_only: bool):
    keep = [i for i, l in enumerate(labels) if (len(l) == 1 or not single_only)]
    cats = sorted({c for i in keep for c in labels[i]} - {neutral})
    neu = [i for i in keep if labels[i] == [neutral]]
    m0 = H[neu].mean(0)
    D, names = [], []
    for c in cats:
        idx = [i for i in keep if c in labels[i]]
        if len(idx) >= min_n:
            D.append(unit(H[idx].mean(0) - m0))
            names.append(c)
    return np.stack(D), names


def top_pcs(D: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the stacked unit category directions (uncentered, as P2 pc1)."""
    return np.linalg.svd(D, full_matrices=False)[2][:k].T


def cmd_cat(a):
    texts, labels = read_labeled(a.data, a.text_col, a.label_col, a.label_sep)
    tok, model = load(a.model, a.revision)
    H = hidden(tok, model, texts, a.layer).astype(np.float64)
    D, names = category_dirs(H, labels, a.neutral, a.min_n, not a.multi_label)
    save_part(a.out, D, names, ["cat"] * len(names), {"cmd": "cat", "data": a.data, "layer": a.layer, "n": len(texts)},
              H_cat2=top_pcs(D, 2), H_cat3=top_pcs(D, 3))


# ------------------------------------------------------------------ Sun et al. VA (H_VA^Sun)
# Re-implementation of the valence/arousal recovery of Sun et al. (arXiv 2604.03147), as verified from the paper.
# Everything the paper specifies is marked [Sun]. Everything it leaves open is marked
# [OURS] = "not specified by Sun; our choice, to be preregistered".
GOEMO_LABELS = ("admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
                "remorse", "sadness", "surprise")          # [Sun] the 27 GoEmotions emotion labels
GOEMO_NEUTRAL = "neutral"                                   # [Sun] the baseline class

# [Sun] the three self-report templates, verbatim (lines joined with "\n").
SUN_TEMPLATES = (
    "\n".join([
        'Rate the emotion label "{label}" on two continuous scales.',
        "Return ONLY a JSON object with numeric fields:",
        '{{"valence": <number>, "arousal": <number>}}',
        "Scale definitions (BOTH inclusive):",
        "- valence in [-1.00, +1.00]: -1.00 very unpleasant, +1.00 very pleasant",
        "- arousal in [-1.00, +1.00]: -1.00 very calm/deactivated, +1.00 very activated/intense",
        "Constraints:",
        "- Use decimals with at most 2 digits after the decimal.",
        "- Values must be within the ranges exactly (inclusive)."]),
    "\n".join([
        "You are scoring affective properties of emotion words on [-1, +1] scales.",
        'Emotion: "{label}"',
        "Valence (pleasantness):",
        "-1.00 = extremely unpleasant, 0.00 = neutral/mixed, +1.00 = extremely pleasant",
        "Arousal (activation/intensity):",
        "-1.00 = very calm/deactivated, 0.00 = neutral, +1.00 = very activated/intense",
        'Return ONLY JSON: {{"valence": x, "arousal": y}}',
        "x and y must be in [-1.00, +1.00] inclusive, with at most 2 decimals."]),
    "\n".join([
        'Give your best guess for the affective coordinates of the emotion label "{label}".',
        "Hard constraints (inclusive):",
        "- valence must be between -1.00 and +1.00",
        "- arousal must be between -1.00 and +1.00",
        "- use at most 2 decimals",
        "Return ONLY JSON with keys valence and arousal."]),
)
SUN_K_GRID = tuple(range(2, 11))                            # [OURS] k grid (Sun reports no k)
SUN_LAM_REL = np.logspace(-4, 2, 13)                        # [OURS] lambda grid, relative to s_1^2 of the fold's Z

SUN_CHOICES = {  # written into the output meta so that the registered values are explicit
    "steering_input": "raw text, no chat template, last token, hidden_states[L] (P3-K §9 convention) [OURS]",
    "steering_vector_norm": "raw mean difference, not unit-normalized before PCA [OURS]",
    "goemotions_rows": "raw config, one row per rater annotation; rows with exactly one positive label kept; "
                       "duplicate texts across raters kept as separate rows [OURS for the duplicate handling]",
    "self_report_format": "chat template (instruct model, generation task), greedy, max_new_tokens=64 [OURS]",
    "self_report_parse": "first {...} JSON object; else regex for valence/arousal numbers; values clipped to "
                         "[-1, 1]; failed templates dropped, label mean over the parsed templates; a label "
                         "with no parsed template is excluded from the fit [OURS]",
    "k_lambda": "chosen jointly per target (V, A) by leave-one-label-out CV over the 27 labels, criterion = "
                "Pearson r of the held-out predictions; ties -> smaller k, then larger lambda [OURS]",
    "lambda_grid": "s_1(Z_train)^2 * logspace(-4, 2, 13) [OURS]",
    "gram_schmidt_order": "V first, A orthogonalized against V [OURS]",
    "nrc_targets": "NRC-VAD valence/arousal of the label word itself; labels missing from the lexicon are "
                   "excluded from that fit [OURS]",
}


def load_goemotions_raw(data: str, max_per_label: int, seed: int) -> tuple[list[str], list[int]]:
    """GoEmotions raw config: one row per (text, rater); one 0/1 column per label. data = 'hf' loads
    google-research-datasets/go_emotions (config 'raw', Apache-2.0) via `datasets`; otherwise a LOCAL CSV/TSV with the
    same columns (text + 28 label columns). Returns texts and label indices (0..26 emotions, 27 = neutral) for the rows
    with exactly one positive label [Sun]."""
    names = list(GOEMO_LABELS) + [GOEMO_NEUTRAL]
    if data == "hf":
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
        cols = {n: np.asarray(ds[n], int) for n in names}
        texts_all = list(ds["text"])
    else:
        delim = "\t" if data.endswith(".tsv") else ","
        rows = list(csv.DictReader(open(data, encoding="utf-8"), delimiter=delim))
        cols = {n: np.array([int(float(r[n])) for r in rows]) for n in names}
        texts_all = [r["text"] for r in rows]
    M = np.stack([cols[n] for n in names], 1)
    keep = np.flatnonzero(M.sum(1) == 1)
    lab = M[keep].argmax(1)
    if max_per_label:                                       # [OURS] optional subsampling (default: all rows)
        rng = np.random.default_rng(seed)
        sel = []
        for c in range(len(names)):
            idx = keep[lab == c]
            sel += sorted(rng.choice(idx, min(max_per_label, len(idx)), replace=False).tolist())
        keep = np.array(sorted(sel))
        lab = M[keep].argmax(1)
    return [texts_all[i] for i in keep], lab.tolist()


def label_sums_multi(tok, model, texts: list[str], lab: list[int], layers: list[int], n_lab: int, bs: int = 32):
    """Streaming per-label sums of the last-token hidden state at several layers (one forward pass per batch).
    Returns sums (n_layers, n_lab, h) float64 and counts (n_lab,)."""
    import torch
    sums, cnt = None, np.zeros(n_lab, int)
    # sort by length to cut padding (sums do not depend on the order); labels are permuted with the texts
    order = np.argsort([len(t) for t in texts], kind="stable")
    texts, lab = [texts[i] for i in order], np.asarray(lab)[order]
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            enc = tok(texts[b:b + bs], return_tensors="pt", padding=True).to("cuda")
            # base model only: same hidden_states, no vocabulary logits (memory on the shared GPU)
            hs = model.model(**enc, output_hidden_states=True).hidden_states
            last = enc["attention_mask"].sum(1) - 1
            ar = torch.arange(last.shape[0], device=last.device)
            H = np.stack([hs[L][ar, last].float().cpu().numpy() for L in layers])   # (nL, B, h)
            if sums is None:
                sums = np.zeros((len(layers), n_lab, H.shape[-1]))
            lb = lab[b:b + bs]
            for c in np.unique(lb):
                sums[:, c] += H[:, lb == c].sum(1)
            cnt += np.bincount(lb, minlength=n_lab)
            if (b // bs) % 200 == 0:
                print(f"  sums {b}/{len(texts)}", flush=True)
    return sums, cnt


def steering_from_sums(sums: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """[Sun] v_e = mean(emotion e) - mean(neutral). sums (..., 28, h) with neutral last -> (..., 27, h)."""
    means = sums / cnt[:, None]
    return means[..., :-1, :] - means[..., -1:, :]


_NUM = r"([-+]?\d*\.?\d+)"


def parse_va(text: str) -> tuple[float, float] | None:
    """[OURS] parse {valence, arousal} from a generation: the first JSON object, else a key: number regex.
    Values are clipped to [-1, 1]; None if either value is missing."""
    import re
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0))
            o = {str(k).lower(): v for k, v in o.items()}
            v, a = float(o["valence"]), float(o["arousal"])
            return float(np.clip(v, -1, 1)), float(np.clip(a, -1, 1))
        except (ValueError, KeyError, TypeError):
            continue
    mv = re.search(r"valence[^-+\d\w]{0,4}" + _NUM, text, re.I)
    ma = re.search(r"arousal[^-+\d\w]{0,4}" + _NUM, text, re.I)
    if mv and ma:
        return float(np.clip(float(mv.group(1)), -1, 1)), float(np.clip(float(ma.group(1)), -1, 1))
    return None


def self_report(tok, model, labels: tuple[str, ...], max_new_tokens: int = 64) -> dict:
    """[Sun] three templates per label, averaged. [OURS] chat template + greedy decoding, one prompt at a time."""
    import torch
    out = {}
    for lab in labels:
        rec = {"raw": [], "parsed": []}
        for tpl in SUN_TEMPLATES:
            msgs = [{"role": "user", "content": tpl.format(label=lab)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                     max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            rec["raw"].append(txt)
            rec["parsed"].append(parse_va(txt))
        ok = [p for p in rec["parsed"] if p is not None]
        rec["n_parsed"] = len(ok)
        rec["valence"] = float(np.mean([p[0] for p in ok])) if ok else None
        rec["arousal"] = float(np.mean([p[1] for p in ok])) if ok else None
        out[lab] = rec
        print(lab, rec["valence"], rec["arousal"], rec["n_parsed"], flush=True)
    return out


def sun_fit(V: np.ndarray, y: np.ndarray, k: int, lam: float, lam_rel: bool = True) -> tuple[np.ndarray, dict]:
    """[Sun] V (n x h) steering vectors, mean-centered; PCA -> U_k (h x k), scores Z = V_c U_k;
    beta = argmin |Z beta - y_c|^2 + lam |beta|^2; w = unit(U_k beta). lam_rel: lam is relative to s_1(Z)^2 [OURS]."""
    mu = V.mean(0)
    Vc = V - mu
    _, s, Wt = np.linalg.svd(Vc, full_matrices=False)
    Uk = Wt[:k].T
    Z = Vc @ Uk
    lam_abs = lam * s[0] ** 2 if lam_rel else lam
    yc = y - y.mean()
    beta = np.linalg.solve(Z.T @ Z + lam_abs * np.eye(k), Z.T @ yc)
    w = Uk @ beta
    return w / np.linalg.norm(w), {"mu": mu, "Uk": Uk, "beta": beta, "lam_abs": float(lam_abs), "y_mean": y.mean(),
                                   "w_raw": w}


def sun_lolo(V: np.ndarray, y: np.ndarray, ks=SUN_K_GRID, lams=SUN_LAM_REL) -> dict:
    """[OURS] leave-one-label-out CV over the labels for (k, lambda). Prediction for the held-out label:
    (v - mu_train) U_k beta + mean(y_train). Criterion: Pearson r of the held-out predictions with y."""
    n = len(y)
    pred = np.zeros((len(ks), len(lams), n))
    for i in range(n):
        tr = np.arange(n) != i
        Vt, yt = V[tr], y[tr]
        mu = Vt.mean(0)
        _, s, Wt = np.linalg.svd(Vt - mu, full_matrices=False)
        yc = yt - yt.mean()
        for a, k in enumerate(ks):
            k_eff = min(k, len(s))
            Uk = Wt[:k_eff].T
            Z = (Vt - mu) @ Uk
            zi = (V[i] - mu) @ Uk
            for b, lr in enumerate(lams):
                beta = np.linalg.solve(Z.T @ Z + lr * s[0] ** 2 * np.eye(k_eff), Z.T @ yc)
                pred[a, b, i] = zi @ beta + yt.mean()
    R = np.array([[np.corrcoef(pred[a, b], y)[0, 1] if np.std(pred[a, b]) > 0 else -1.0
                   for b in range(len(lams))] for a in range(len(ks))])
    R = np.nan_to_num(R, nan=-1.0)
    best = max(((a, b) for a in range(len(ks)) for b in range(len(lams))),
                key=lambda ab: (round(R[ab], 10), -ks[ab[0]], lams[ab[1]]))
    return {"k": int(ks[best[0]]), "lam_rel": float(lams[best[1]]), "cv_r": float(R[best]),
            "cv_pred": pred[best].tolist(), "cv_table": R.tolist(), "k_grid": list(ks), "lam_grid": lams.tolist()}


def gram_schmidt_va(w_v: np.ndarray, w_a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """[Sun] orthonormalize; [OURS] V first. Returns (v, a, cos(w_v, w_a) before orthogonalization)."""
    v = w_v / np.linalg.norm(w_v)
    c = float(v @ (w_a / np.linalg.norm(w_a)))
    a = w_a - (w_a @ v) * v
    return v, a / np.linalg.norm(a), c


def sun_axes(V: np.ndarray, yv: np.ndarray, ya: np.ndarray) -> dict:
    """Full H_VA^Sun pipeline on one layer for one target set. Rows with NaN targets are excluded per target.
    Recovery: r(projection of the centered steering vectors on the final axis, target), in-sample, plus the LOLO CV r
    of the selected (k, lambda) (the honest, out-of-sample version)."""
    res, W = {}, {}
    for nm, y in (("V", yv), ("A", ya)):
        m = ~np.isnan(y)
        cv = sun_lolo(V[m], y[m])
        w, fit = sun_fit(V[m], y[m], cv["k"], cv["lam_rel"])
        W[nm] = w
        res[nm] = {"n_labels": int(m.sum()), "k": cv["k"], "lam_rel": cv["lam_rel"], "lam_abs": fit["lam_abs"],
                   "cv_r": cv["cv_r"], "cv_table": cv["cv_table"], "k_grid": cv["k_grid"], "lam_grid": cv["lam_grid"],
                   "mask": m}
    v, a, c = gram_schmidt_va(W["V"], W["A"])
    Vc = V - V.mean(0)
    for nm, ax, y in (("V", v, yv), ("A", a, ya)):
        m = res[nm].pop("mask")
        proj = Vc @ ax
        res[nm]["r_insample"] = float(np.corrcoef(proj[m], y[m])[0, 1])
        res[nm]["proj"] = proj.tolist()
    res["cos_wV_wA_before_GS"] = c
    return {"v": v, "a": a, "fit": res}


def _ratings_from_report(rep: dict) -> tuple[np.ndarray, np.ndarray]:
    yv = np.array([np.nan if rep[l]["valence"] is None else rep[l]["valence"] for l in GOEMO_LABELS])
    ya = np.array([np.nan if rep[l]["arousal"] is None else rep[l]["arousal"] for l in GOEMO_LABELS])
    return yv, ya


def _nrc_targets(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    words, Y = read_ratings(path, "nrc", ["V", "A"], None, None)
    lut = {w.lower(): y for w, y in zip(words, Y)}
    y = np.array([lut.get(l, [np.nan, np.nan]) for l in GOEMO_LABELS], float)
    return y[:, 0], y[:, 1], {"missing_labels": [l for l in GOEMO_LABELS if l not in lut]}


def cmd_sun_va(a):
    """H_VA^Sun (Sun et al., arXiv 2604.03147). Steps:
      1. GoEmotions raw, rows with exactly one label; last-token hidden states of the raw text at every layer in
         --check_layers (and --layer); steering vector v_e = mean(e) - mean(neutral) for the 27 emotions.
      2. Self-report: three verbatim templates per label, greedy, JSON {valence, arousal}, mean over templates
         (cached in --ratings; reused if the file exists).
      3. Per layer: mean-center, PCA, ridge on the PC scores (k and lambda by leave-one-label-out CV), w = U_k beta,
         Gram-Schmidt (V first). Optional NRC-VAD-supervised variant (--nrc LOCAL path).
      4. Reproduction check: label-level recovery r per layer (in-sample projection r and LOLO CV r).
         Sun's reported values for Llama-3.1-8B: valence r ~ .97, arousal r ~ .87 (self-report) / .95 (NRC-VAD).
      5. Optional word-level check (--word_check N): N NRC words (raw text), projection on the axes vs NRC V/A.
    Output (--out): D = [sun_V, sun_A], groups va_sun, H_VA_sun at --layer. NRC variant (--out_nrc): sun_nrc_V/A,
    group va_sun_nrc, H_VA_sun_nrc (not in the default affect groups; reproduction/sensitivity only).
    Side files: <out>.report.json (all fits, recovery per layer, ratings, choices) and <out>.vectors.npz
    (steering vectors per layer, so that refits run on the CPU)."""
    if a.selftest:
        return sun_selftest(a.reps)
    for req in ("layer", "out"):
        assert getattr(a, req) is not None, f"--{req} is required"
    layers = sorted({a.layer} | {int(x) for x in a.check_layers.split(",") if x})
    _m = {}

    def tm():  # the model is loaded only when needed: with cached vectors and ratings the refit runs on the CPU
        if not _m:
            assert a.model and a.revision, "--model and --revision are required (no cached vectors/ratings)"
            _m["tm"] = load(a.model, a.revision)
        return _m["tm"]
    stem = a.out[:-4] if a.out.endswith(".npz") else a.out
    vec_path = stem + ".vectors.npz"
    if Path(vec_path).exists():                              # resume: steering vectors already computed
        z = np.load(vec_path)
        assert list(z["layers"]) == layers, "cached vectors were computed for other layers"
        Vall, cnt = z["V"], z["counts"]
    else:
        texts, lab = load_goemotions_raw(a.data, a.max_per_label, SEED["subsample"])
        print(json.dumps({"goemotions_rows_single_label": len(texts)}), flush=True)
        sums, cnt = label_sums_multi(*tm(), texts, lab, layers, len(GOEMO_LABELS) + 1, a.bs)
        Vall = steering_from_sums(sums, cnt)
        np.savez(vec_path, V=Vall, counts=cnt, layers=np.array(layers), labels=np.array(GOEMO_LABELS))
    if Path(a.ratings).exists():
        rep = json.load(open(a.ratings, encoding="utf-8"))
    else:
        rep = self_report(*tm(), GOEMO_LABELS)
        Path(a.ratings).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    yv, ya = _ratings_from_report(rep)
    targets = {"self": (yv, ya, {"n_unparsed_labels": int(np.isnan(yv).sum())})}
    if a.nrc:
        targets["nrc"] = _nrc_targets(a.nrc)
    report = {"choices": SUN_CHOICES, "layers": layers, "main_layer": a.layer, "counts": cnt.tolist(),
              "labels": list(GOEMO_LABELS), "fits": {}, "sun_reported_llama31_8b":
                  {"self": {"V": 0.97, "A": 0.87}, "nrc": {"A": 0.95}}}
    axes = {}
    for tname, (ty_v, ty_a, tinfo) in targets.items():
        report["fits"][tname] = {"target_info": tinfo, "targets_V": np.nan_to_num(ty_v, nan=-9).tolist(),
                                 "targets_A": np.nan_to_num(ty_a, nan=-9).tolist(), "per_layer": {}}
        for li, L in enumerate(layers):
            r = sun_axes(Vall[li], ty_v, ty_a)
            report["fits"][tname]["per_layer"][str(L)] = r["fit"]
            if L == a.layer:
                axes[tname] = (r["v"], r["a"])
            print(tname, L, {d: (r["fit"][d]["r_insample"], r["fit"][d]["cv_r"], r["fit"][d]["k"])
                             for d in ("V", "A")}, flush=True)
    if a.word_check and a.nrc:
        words, Y = read_ratings(a.nrc, "nrc", ["V", "A"], None, None)
        idx = np.sort(np.random.default_rng(SEED["subsample"]).choice(len(words), min(a.word_check, len(words)),
                                                                     replace=False))
        Hw = hidden(*tm(), [words[i] for i in idx], a.layer).astype(np.float64)
        report["word_check"] = {}
        for tname, (v, ax_a) in axes.items():
            report["word_check"][tname] = {"n_words": len(idx),
                                           "r_V": float(np.corrcoef(Hw @ v, Y[idx, 0])[0, 1]),
                                           "r_A": float(np.corrcoef(Hw @ ax_a, Y[idx, 1])[0, 1])}
    Path(stem + ".report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    meta = {"cmd": "sun-va", "layer": a.layer, "data": a.data, "ratings": a.ratings, "choices": SUN_CHOICES,
            "fit": {d: {kk: report["fits"]["self"]["per_layer"][str(a.layer)][d][kk]
                        for kk in ("k", "lam_rel", "cv_r", "r_insample", "n_labels")} for d in ("V", "A")}}
    v, ax_a = axes["self"]
    save_part(a.out, [v, ax_a], ["sun_V", "sun_A"], ["va_sun", "va_sun"], meta,
              H_VA_sun=orth_basis(np.stack([v, ax_a], 1)))
    if "nrc" in axes and a.out_nrc:
        v, ax_a = axes["nrc"]
        save_part(a.out_nrc, [v, ax_a], ["sun_nrc_V", "sun_nrc_A"], ["va_sun_nrc", "va_sun_nrc"],
                  dict(meta, cmd="sun-va (NRC-VAD supervised)", nrc=a.nrc),
                  H_VA_sun_nrc=orth_basis(np.stack([v, ax_a], 1)))


def sun_selftest(reps: int = 5) -> None:
    """Synthetic check of the CPU part of sun-va: planted 2-D (V, A) structure in example-level hidden states,
    hidden inside larger nuisance structure and noise. Checks recovery of the planted axes, r, the CV choice,
    the null (random targets), the steering-vector arithmetic and the JSON parser."""
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    # parser
    cases = {'{"valence": -0.8, "arousal": 0.65}': (-0.8, 0.65), 'Sure! ```json\n{"valence": 0.70, "arousal": 1.4}```':
             (0.7, 1.0), "valence: -0.3, arousal: 0.2": (-0.3, 0.2), "I cannot rate that.": None,
             '{"Valence": "0.5", "Arousal": "-0.25"}': (0.5, -0.25)}
    got = {s: parse_va(s) for s in cases}
    check("parse_va", all(got[s] == cases[s] for s in cases), got)
    check("templates", all("{label}" not in t.format(label="joy") and '"joy"' in t.format(label="joy")
                           for t in SUN_TEMPLATES) and SUN_TEMPLATES[0].format(label="x").count("\n") == 8,
          "3 templates render; T1 has 9 lines")
    h, n_lab = 256, len(GOEMO_LABELS)
    res = {"cos_V": [], "cos_A": [], "rV": [], "rA": [], "cvV": [], "cvA": [], "kV": [], "kA": [], "null_cv": [],
           "orth": [], "nrc_like": []}
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        Q = np.linalg.qr(rng.standard_normal((h, 12)))[0]
        aV, aA, nuis = Q[:, 0], Q[:, 1], Q[:, 2:8]
        tv, ta = rng.uniform(-1, 1, n_lab), rng.uniform(-1, 1, n_lab)            # planted label coordinates
        # label means: V/A structure (scale 2) + larger nuisance structure (scale 3, 6 dims, so V/A is NOT the top
        # PC pair) + label-specific isotropic noise
        mu_lab = (2.0 * np.outer(tv, aV) + 2.0 * np.outer(ta, aA) + 3.0 * rng.standard_normal((n_lab, 6)) @ nuis.T
                  + 0.05 * rng.standard_normal((n_lab, h)))
        mu_neu = rng.standard_normal(h)
        # example-level hidden states (per-example isotropic noise, shared offset); neutral class last
        n_per = rng.integers(150, 400, n_lab + 1)
        lab = np.repeat(np.arange(n_lab + 1), n_per)
        mean_rows = np.vstack([mu_neu + mu_lab, mu_neu[None]])
        X = mean_rows[lab] + 1.0 * rng.standard_normal((len(lab), h)) + 5.0 * rng.standard_normal(h)
        sums = np.zeros((n_lab + 1, h))
        np.add.at(sums, lab, X)
        cnt = np.bincount(lab)
        V = steering_from_sums(sums, cnt)
        if rep == 0:
            direct = np.stack([X[lab == c].mean(0) - X[lab == n_lab].mean(0) for c in range(n_lab)])
            check("steering_from_sums", np.allclose(V, direct), "sum/count arithmetic equals direct means")
        # self-report ratings: planted coordinates + rating noise; arousal noisier (as in Sun: .87 < .97)
        yv = np.clip(tv + 0.1 * rng.standard_normal(n_lab), -1, 1)
        ya = np.clip(ta + 0.25 * rng.standard_normal(n_lab), -1, 1)
        yv[3] = np.nan                                       # one unparsed label must be handled
        r = sun_axes(V, yv, ya)
        res["cos_V"].append(abs(float(r["v"] @ aV)))
        res["cos_A"].append(abs(float(r["a"] @ aA)))
        res["rV"].append(r["fit"]["V"]["r_insample"])
        res["rA"].append(r["fit"]["A"]["r_insample"])
        res["cvV"].append(r["fit"]["V"]["cv_r"])
        res["cvA"].append(r["fit"]["A"]["cv_r"])
        res["kV"].append(r["fit"]["V"]["k"])
        res["kA"].append(r["fit"]["A"]["k"])
        res["orth"].append(abs(float(r["v"] @ r["a"])))
        # null: targets unrelated to the vectors
        res["null_cv"].append(sun_axes(V, rng.uniform(-1, 1, n_lab), rng.uniform(-1, 1, n_lab))["fit"]["V"]["cv_r"])
        # NRC-like targets on a different scale (0..1) and a missing label: direction must be the same
        rn = sun_axes(V, (yv + 1) / 2, (ya + 1) / 2)
        res["nrc_like"].append(abs(float(rn["v"] @ aV)))
    f = lambda x: [round(float(v), 3) for v in x]
    check("planted V axis recovered", min(res["cos_V"]) > 0.9, f"|cos| {f(res['cos_V'])}")
    check("planted A axis recovered", min(res["cos_A"]) > 0.8, f"|cos| {f(res['cos_A'])}")
    check("label-level recovery r (in-sample)", min(res["rV"]) > 0.9 and min(res["rA"]) > 0.75,
          f"r_V {f(res['rV'])} r_A {f(res['rA'])}")
    check("LOLO CV r", min(res["cvV"]) > 0.85 and min(res["cvA"]) > 0.6, f"cv_V {f(res['cvV'])} cv_A {f(res['cvA'])}")
    check("chosen k (V/A planted below 6 larger nuisance PCs: k must go past them; median >= 8, all >= 7)",
          np.median(res["kV"] + res["kA"]) >= 8 and min(res["kV"] + res["kA"]) >= 7,
          f"k_V {res['kV']} k_A {res['kA']}")
    check("Gram-Schmidt orthonormal", max(res["orth"]) < 1e-10, f"max |v.a| {max(res['orth']):.1e}")
    check("null targets: LOLO CV r low", np.median(res["null_cv"]) < 0.4, f"cv_r {f(res['null_cv'])}")
    check("scale-free targets (NRC-like 0..1, one missing)", min(res["nrc_like"]) > 0.9, f"|cos| {f(res['nrc_like'])}")
    print("SUN_SELFTEST_OK" if ok else "SUN_SELFTEST_FAILED")


# ------------------------------------------------------------------ variance-matched random directions (P2)
def cmd_random(a):
    """n random directions per target, each matched to the target's variance on the activation covariance of
    p3k_gradients.py (act_stats.npz). Seed fixed per target index."""
    S = np.load(a.act_stats)
    cov, n_rows = S["cov"].astype(np.float64), int(S["n"])
    evals, evecs = np.linalg.eigh(cov)
    T = np.load(a.targets)
    D, names = [], []
    for i, (v, nm) in enumerate(zip(T["D"], T["names"])):
        if a.groups and str(T["groups"][i]) not in a.groups.split(","):
            continue
        tv = float(v @ cov @ v) * (n_rows - 1) / n_rows
        rs = fast_var_matched(np.random.default_rng(SEED["rand"] + i), evals, evecs, n_rows, tv, a.n)
        for j, r in enumerate(rs):
            D.append(r)
            names.append(f"rand_{nm}_{j}")
    save_part(a.out, D, names, ["rand"] * len(names), {"cmd": "random", "targets": a.targets, "n_per_target": a.n})


# ------------------------------------------------------------------ assemble
def cmd_assemble(a):
    D, names, groups, subs, meta = [], [], [], {}, []
    for p in a.parts:
        z = np.load(p, allow_pickle=False)
        D.append(z["D"])
        names += [str(x) for x in z["names"]]
        groups += [str(x) for x in z["groups"]]
        meta.append(json.loads(str(z["meta"])))
        for k in z.files:
            if k.startswith("H_"):
                assert k not in subs, f"duplicate subspace {k}"
                subs[k] = z[k]
    assert len(set(names)) == len(names), "duplicate direction names"
    save_part(a.out, np.concatenate(D), names, groups, {"cmd": "assemble", "parts": meta}, **subs)


# ------------------------------------------------------------------ layer selection (no task gradients)
def cmd_select_layer(a):
    """For each candidate layer: category directions from two random halves of the items (n_rep times); score = mean
    over categories of cos(half a, half b). The layer with the highest score is chosen (prereg §2: separation of the
    emotion directions only, never the task gradients)."""
    texts, labels = read_labeled(a.data, a.text_col, a.label_col, a.label_sep)
    tok, model = load(a.model, a.revision)
    rng = np.random.default_rng(SEED["layer_split"])
    res = {}
    for L in [int(x) for x in a.layers.split(",")]:
        H = hidden(tok, model, texts, L).astype(np.float64)
        sc = []
        for _ in range(a.n_rep):
            m = rng.permutation(len(texts)) < len(texts) // 2
            Da, na = category_dirs(H[m], [labels[i] for i in np.flatnonzero(m)], a.neutral, a.min_n, True)
            Db, nb = category_dirs(H[~m], [labels[i] for i in np.flatnonzero(~m)], a.neutral, a.min_n, True)
            common = sorted(set(na) & set(nb))
            sc.append(float(np.mean([Da[na.index(c)] @ Db[nb.index(c)] for c in common])))
        res[L] = {"split_cos_mean": float(np.mean(sc)), "split_cos_sd": float(np.std(sc))}
        print(L, res[L], flush=True)
    best = max(res, key=lambda k: res[k]["split_cos_mean"])
    Path(a.out).write_text(json.dumps({"layers": res, "chosen": best, "data": a.data}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    def gpu(p):
        p.add_argument("--model", required=True)
        p.add_argument("--revision", required=True)
        p.add_argument("--layer", type=int, required=p.prog.split()[-1] != "select-layer")
        p.add_argument("--out", required=True)

    p = sp.add_parser("contrast")
    gpu(p)
    p.add_argument("--spec", required=True)
    p.add_argument("--splits", default="disc")
    p.add_argument("--sets", default="", help="minimal-pair spec only: set names or groups to keep (default all)")
    p = sp.add_parser("neutral-pcs")
    gpu(p)
    p.add_argument("--corpus", required=True, help="LOCAL neutral corpus, one sentence per line")
    here = Path(__file__).resolve().parent             # flat server layout: stimuli_v0 next to this file
    p.add_argument("--stim_pairs", default=str(next((d / "stimuli_v0" / "p3k_pairs.csv" for d in (here, here.parent)
                                                     if (d / "stimuli_v0").is_dir()), "stimuli_v0/p3k_pairs.csv")))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min_words", type=int, default=5)
    p.add_argument("--max_words", type=int, default=40)
    p = sp.add_parser("vad-ext")
    gpu(p)
    p.add_argument("--data", required=True, help="LOCAL ratings file (not redistributed)")
    p.add_argument("--format", choices=("nrc", "emobank", "csv"), required=True)
    p.add_argument("--dims", default="V,A")
    p.add_argument("--text_col", default=None)
    p.add_argument("--cols", default=None)
    p.add_argument("--template", default="{text}")
    p.add_argument("--name", default="ext")
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--save_pattern", action="store_true")
    for nm in ("cat", "select-layer"):
        p = sp.add_parser(nm)
        gpu(p)
        p.add_argument("--data", required=True, help="LOCAL labeled CSV/TSV")
        p.add_argument("--text_col", default="text")
        p.add_argument("--label_col", default="label")
        p.add_argument("--label_sep", default=",")
        p.add_argument("--neutral", default="neutral")
        p.add_argument("--min_n", type=int, default=30)
        if nm == "cat":
            p.add_argument("--multi_label", action="store_true")
        else:
            p.add_argument("--layers", required=True, help="candidate layers near Sun's VA maximum, e.g. 10,12,14,16")
            p.add_argument("--n_rep", type=int, default=20)
    p = sp.add_parser("sun-va")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--layer", type=int, help="hidden_states index of the output axes")
    p.add_argument("--out")
    p.add_argument("--data", default="hf", help="'hf' (google-research-datasets/go_emotions, raw) or LOCAL CSV/TSV")
    p.add_argument("--check_layers", default="", help="extra layers for the reproduction check, e.g. 8,10,12,14,16")
    p.add_argument("--ratings", default="sun_self_report.json", help="self-report cache (created if missing)")
    p.add_argument("--nrc", default="", help="LOCAL NRC-VAD file (not redistributed); enables the NRC variant")
    p.add_argument("--out_nrc", default="")
    p.add_argument("--word_check", type=int, default=0, help="n NRC words for the optional word-level check")
    p.add_argument("--max_per_label", type=int, default=0, help="0 = all single-label rows")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--selftest", action="store_true", help="synthetic CPU check (no model)")
    p.add_argument("--reps", type=int, default=5)
    p = sp.add_parser("random")
    p.add_argument("--act_stats", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--groups", default="")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sp.add_parser("assemble")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"contrast": cmd_contrast, "vad-ext": cmd_vad_ext, "cat": cmd_cat, "sun-va": cmd_sun_va, "random": cmd_random,
     "assemble": cmd_assemble, "select-layer": cmd_select_layer, "neutral-pcs": cmd_neutral_pcs}[a.cmd](a)


if __name__ == "__main__":
    main()
