"""P3-K step 1 (GPU): readout gradients, shared judgment gradient g_t, item-split halves, and no-steering validity.
Plan: 00_研究計画/12_研究計画_v2.2_20261124.md §1, §5; prereg draft: 02_設計/04_事前登録草案_撤退実験_v2_20261124.md §4.
Stimuli: ../stimuli_v1/p3k_readouts_map{1..6}.csv (28 tasks V1-V16, N1-N12 x 1,020 rows; C1-C4 baselines; maps 1-4
byte-identical to stimuli_v0, map 5 = Ж Щ Ф Ю, map 6 = Д Л ø æ; P2 used maps 1, 2).

Method (copied from P2 s1o_rating.py / body.tex §4.2-4.3, nothing new):
  * prompt = prompt_qa + "\\nAnswer:"; next-token log-probabilities renormalized over the answer tokens (" 1".." 9" or
    " <label>"); score s_i = w_i . log p_i with the P2 weights (digit: OLS slope over k; ranked options: slope over rank,
    4 rotations; pair: log-odds of the heavier option, both orders). The label order differs between items
    (Latin square, stimuli README), so the answer token ids are per item (P2 had one fixed set).
  * gradient of s_i w.r.t. a vector delta added to the output of decoder layer L-1 (= hidden_states[L]) at every position
    (P2 Steer hook; --grad_pos last adds it at the last token only), delta = 0, fp32. Per-item gradients are summed per
    scenario (both severities, all rotations), so that any scenario split can be recomputed on the CPU.
  * shared judgment gradient (p3k_common.shared_gradient): judgment gradients (J, JR, CAT, PAIR) residualized on the span
    of the control gradients (CTRL, CTRL_CAT), JR sign-reversed, averaged without normalization.
  * sigma: n_splits random 2-splits of the 30 scenarios (seeded); shared gradient of each half (control span re-estimated
    within the half). sigma itself needs P_A and is computed in p3k_kill_tests.py.
  * validity without steering (P2 §4.2 criteria, p3k_common.validity_readout) + the gradient positive control
    (unit gradient from the odd scenarios, applied at alpha, effect on the even scenarios vs n_null variance-matched random
    directions, one-sided p < .05). The emotion directions are never applied here.
Units (resumable): (task, 'digit') for J, JR, CTRL (identical in all mappings) and (task, map m) for CAT, PAIR, CTRL_CAT.
Outputs in --out/<model>_L<layer>/: act_stats.npz, unembed.npz, unit_<task>_<digit|mapM>.{npz,json},
shared_<task>_map<M>.npz (+ .json with task entry), base_<C1..C4>.npz (exploratory baselines), run_<...>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from p3k_common import (CTRL, DIGIT_R, JUDG, LABEL_R, READ, assemble_shared, base_scores, baseline_weights,
                        canonical_items, fast_var_matched, scen_num, score_weights, task_entry, unit, validity_readout)

SEED = {"split": 20261201, "null": 20261202, "fd": 20261203}
TASKS = tuple([f"V{i}" for i in range(1, 17)] + [f"N{i}" for i in range(1, 13)])
BASELINES = ("C1", "C2", "C3", "C4")


# ------------------------------------------------------------------ model helpers (P2 s1_light_round.py)
class Steer:
    """Adds a vector to the output of one decoder layer (at every position, or where a position mask selects)."""

    def __init__(self, layer: torch.nn.Module):
        self.v = None
        self.h = layer.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.v is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.v.to(out[0].dtype),) + tuple(out[1:])
        return out + self.v.to(out.dtype)


def decoder_layers(model):
    for path in (("model", "layers"), ("model", "language_model", "layers")):
        obj = model
        try:
            for p in path:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found")


def load_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


class Readout:
    """Items of one readout, batched in order of length, with per-item answer token ids and score weights
    (generalizes P2 s1o_rating.Readout to per-item label orders)."""

    def __init__(self, tok, items, choices: list[list[str]], W: np.ndarray, last_only: bool = False):
        seqs, cids = [], []
        for it, ch in zip(items, choices):
            full = [tok(it["prompt"] + f" {k}")["input_ids"] for k in ch]
            pre = full[0][:-1]
            assert all(f[:-1] == pre for f in full), it["item_id"]       # every answer is one token after the prefix
            cids.append([f[-1] for f in full])
            seqs.append(pre)
        self.order = np.argsort([len(s) for s in seqs], kind="stable")
        self.seqs = [seqs[i] for i in self.order]
        self.cids = torch.tensor(np.asarray(cids)[self.order]).cuda()
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.W = W
        self.items = items
        self.scen = np.array([x["scen"] for x in items])
        self.even = np.array([scen_num(x["scen"]) % 2 == 0 for x in items])
        self.last_only = last_only

    def _batch(self, pos):
        s = [self.seqs[i] for i in pos]
        T = max(len(x) for x in s)
        ids = torch.full((len(s), T), self.pad, dtype=torch.long)
        mask = torch.zeros((len(s), T), dtype=torch.long)
        for j, x in enumerate(s):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        return ids.cuda(), mask.cuda(), torch.tensor([len(x) - 1 for x in s]).cuda()

    def _vec(self, v, last, T):
        """Steering tensor: v at every position, or only at the last token of each item (--grad_pos last)."""
        if v is None:
            return None
        v = v if v.dim() == 3 else v.reshape(1, 1, -1)
        if not self.last_only:
            return v
        oh = torch.zeros((len(last), T, 1), device="cuda", dtype=torch.float32)
        oh[torch.arange(len(last)), last] = 1.0
        return v * oh

    @torch.no_grad()
    def logp(self, model, steer, v, sel=None, bs: int = 64) -> np.ndarray:
        """log softmax over each item's answer tokens, items in the original order (rows not in sel are NaN)."""
        sel = np.ones(len(self.items), bool) if sel is None else sel
        pos_all = np.flatnonzero(sel[self.order])
        vt = None if v is None else torch.tensor(v, device="cuda", dtype=torch.float32)
        P = np.full((len(self.items), self.cids.shape[1]), np.nan)
        for b in range(0, len(pos_all), bs):
            pos = pos_all[b:b + bs]
            ids, mask, last = self._batch(pos)
            steer.v = self._vec(vt, last, ids.shape[1])
            lg = model(input_ids=ids, attention_mask=mask).logits[torch.arange(len(pos)), last]
            steer.v = None
            lp = torch.log_softmax(torch.gather(lg.float(), 1, self.cids[pos]), -1)
            P[self.order[pos]] = lp.cpu().numpy()
        return P

    def grads(self, model, steer, d: int, bs: int = 8) -> np.ndarray:
        """Per-item d s_i / d delta at delta = 0 (fp32), items in the original order: (n_items, d)."""
        out = np.zeros((len(self.items), d))
        for b in range(0, len(self.seqs), bs):
            pos = np.arange(b, min(b + bs, len(self.seqs)))
            ids, mask, last = self._batch(pos)
            w = torch.tensor(self.W[self.order[pos]], device="cuda", dtype=torch.float32)
            delta = torch.zeros((len(pos), 1, d), device="cuda", dtype=torch.float32, requires_grad=True)
            steer.v = self._vec(delta, last, ids.shape[1])
            with torch.enable_grad():
                lg = model(input_ids=ids, attention_mask=mask).logits[torch.arange(len(pos)), last]
                obj = (torch.log_softmax(torch.gather(lg.float(), 1, self.cids[pos]), -1) * w).sum()
                obj.backward()
            steer.v = None
            out[self.order[pos]] = delta.grad[:, 0].detach().cpu().numpy().astype(np.float64)
            del lg, obj, delta
        return out


@torch.no_grad()
def last_hidden_batched(model, tok, texts, L, bs: int = 32) -> np.ndarray:
    """hidden_states[L] at the last token (right padding; causal attention, so padding does not change it)."""
    out = []
    for b in range(0, len(texts), bs):
        enc = tok(texts[b:b + bs], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():                      # base model only: same hidden_states, no vocabulary logits
            hs = model.model(**enc, output_hidden_states=True).hidden_states[L]
        last = enc["attention_mask"].sum(1) - 1
        out.append(hs[torch.arange(hs.shape[0]), last].float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True, help="directory with p3k_readouts_map{1..6}.csv (stimuli_v1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=None, help="hidden_states index L (steer at decoder layer L-1)")
    ap.add_argument("--depth", type=float, default=None, help="relative depth (P2: 0.33), used if --layer is absent")
    ap.add_argument("--maps", default="1,2,3,4,5,6")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--baselines", default=",".join(BASELINES), help="exploratory C1-C4 gradients ('' to skip)")
    ap.add_argument("--alpha_frac", type=float, default=0.2)
    ap.add_argument("--n_null", type=int, default=100, help="random directions per readout (positive control)")
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--grad_pos", choices=("all", "last"), default="all")
    ap.add_argument("--act_max", type=int, default=8400, help="prompts for the activation statistics")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_scen", type=int, default=0, help="smoke runs: keep only the first n scenarios per task")
    ap.add_argument("--fd_check", type=int, default=0, help="finite-difference check on n random directions (J)")
    a = ap.parse_args()
    maps = [m for m in a.maps.split(",") if m]
    tasks = [t for t in a.tasks.split(",") if t]

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    nl = cfg.num_hidden_layers if hasattr(cfg, "num_hidden_layers") else cfg.text_config.num_hidden_layers
    L = a.layer if a.layer is not None else int(round(a.depth * nl))
    tag = f"{a.model.split('/')[-1]}_L{L}"
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    run = {"args": vars(a), "layer": L, "seeds": SEED, "t0": time.time(),
           "env": {"torch": torch.__version__, "transformers": transformers.__version__}}
    run_f = out / f"run_{int(run['t0'])}.json"
    save_run = lambda: run_f.write_text(json.dumps(run, indent=1, default=float))
    save_run()

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    steer = Steer(decoder_layers(model)[L - 1])
    hdim = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size

    R = {m: load_csv(Path(a.stim) / f"p3k_readouts_map{m}.csv") for m in sorted(set(maps) | {"1"})}
    for m in R:
        for x in R[m]:
            x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    keep_scen = None
    if a.max_scen:
        keep_scen = {x["scen"] for x in R["1"] if scen_num(x["scen"]) <= a.max_scen}

    def rows(m, task, readout):
        r = [x for x in R[m] if x["ptask"] == task and x["readout"] == readout]
        return [x for x in r if keep_scen is None or x["scen"] in keep_scen]

    # ---------------- activation statistics (metric rule, whitening, variance-matched nulls) and alpha
    fa = out / "act_stats.npz"
    if not fa.exists():
        pr = [x["prompt"] for x in R["1"] if x["ptask"] in TASKS and x["rot"] in ("", "0")
              and x["readout"] in ("J", "JR", "CTRL", "CAT", "CTRL_CAT")]
        pr = pr[: a.act_max]
        H = last_hidden_batched(model, tok, pr, L)
        mu = H.mean(0)
        Hc = H - mu
        cov = Hc.T @ Hc / (len(H) - 1)
        np.savez(fa, mean=mu, mean_abs=np.abs(H).mean(0), cov=cov.astype(np.float32), n=len(H),
                 mean_norm=float(np.linalg.norm(H, axis=1).mean()), layer=L)
        del H, Hc
    A = np.load(fa)
    cov = A["cov"].astype(np.float64)
    n_rows = int(A["n"])
    evals, evecs = np.linalg.eigh(cov)
    alpha = a.alpha_frac * float(A["mean_norm"])
    run["alpha"] = alpha
    save_run()

    # unembedding rows of the answer tokens (exploratory: angle between F and the label-difference space)
    fu = out / "unembed.npz"
    if not fu.exists():
        Wu = model.get_output_embeddings().weight
        ue = {}
        for m in maps:
            labs = sorted({lab for x in R[m] if x["labels"] for lab in x["labels"].split("|")})
            ids = [tok(f"Answer: {lab}")["input_ids"][-1] for lab in labs]
            ue[f"map{m}"] = Wu[ids].float().cpu().numpy()
            ue[f"map{m}_labels"] = np.array(labs)
        ids = [tok(f"Answer: {k}")["input_ids"][-1] for k in range(1, 10)]
        ue["digits"] = Wu[ids].float().cpu().numpy()
        np.savez(fu, **ue)

    def null_dirs(v: np.ndarray, key: str) -> list[np.ndarray]:
        """Variance-matched random directions (P2 procedure), matched to the variance of v; seed from the unit key."""
        tv = float(v @ cov @ v) * (n_rows - 1) / n_rows
        rng = np.random.default_rng(SEED["null"] + zlib.crc32(key.encode()) % 100000)
        return fast_var_matched(rng, evals, evecs, n_rows, tv, a.n_null)

    def run_unit(task: str, unit_name: str, readouts: tuple[str, ...], m: str, ref_J: list[float] | None):
        """Gradients, base scores, validity and positive control for the readouts of one unit."""
        f_npz, f_js = out / f"unit_{task}_{unit_name}.npz", out / f"unit_{task}_{unit_name}.json"
        if f_js.exists() and json.load(open(f_js)).get("done"):
            return json.load(open(f_js))
        t0 = time.time()
        js = {"task": task, "unit": unit_name, "map": m, "readouts": readouts, "layer": L, "alpha": alpha,
              "grad_pos": a.grad_pos}
        arrs = {}
        scens = None
        for t in readouts:
            items = canonical_items(rows(m, task, t), t)
            W, ch = score_weights(items, t)
            ro = Readout(tok, items, ch, W, last_only=a.grad_pos == "last")
            sc_list = sorted({x["scen"] for x in items})
            scens = sc_list if scens is None else scens
            assert sc_list == scens
            n_unit = 2 * len(scens)
            base = ro.logp(model, steer, None)
            bs_ = base_scores(t, base, W, n_unit)
            if t == "J":
                ref_J = bs_["score"]
            s, v = validity_readout(t, bs_, ref_J)
            gi = ro.grads(model, steer, hdim, bs=a.bs)
            idx = np.array([scens.index(x["scen"]) for x in items])
            S = np.zeros((len(scens), hdim))
            np.add.at(S, idx, gi)
            arrs[f"S_{t}"] = S
            arrs[f"N_{t}"] = np.bincount(idx, minlength=len(scens)).astype(float)
            # gradient positive control (necessary condition, P2 §4.2)
            odd = ~ro.even
            dpos = unit(gi[odd].sum(0))
            sco = lambda v_: float(np.nanmean(((ro.logp(model, steer, alpha * v_, ro.even) - base) * W).sum(1)[ro.even]))
            obs = sco(dpos)
            nul = np.array([sco(r) for r in null_dirs(dpos, f"{task}|{unit_name}|{t}")])
            pos_p = float((np.sum(nul >= obs) + 1) / (len(nul) + 1))
            js.setdefault("base", {})[t] = bs_
            js.setdefault("summ", {})[t] = s
            js.setdefault("valid_noSteer", {})[t] = v
            js.setdefault("pos", {})[t] = {"obs": obs, "null_mean": float(nul.mean()), "null_sd": float(nul.std(ddof=1))
                                           if len(nul) > 1 else None, "p": pos_p, "n_null": int(len(nul))}
            js.setdefault("valid", {})[t] = bool(v and pos_p < .05)
            js.setdefault("grad_norm", {})[t] = float(np.linalg.norm(gi.mean(0)))
            arrs[f"null_{t}"] = nul
        arrs["scen"] = np.array(scens)
        np.savez(f_npz, **arrs)
        js["ref_J"] = ref_J
        js["t_s"] = time.time() - t0
        js["done"] = True
        f_js.write_text(json.dumps(js, indent=1, default=float))
        return js

    def assemble(task: str, m: str):
        """Shared judgment gradient of (task, map m), split halves, task entry."""
        fo = out / f"shared_{task}_map{m}.npz"
        if fo.exists():
            return
        D = np.load(out / f"unit_{task}_digit.npz")
        M = np.load(out / f"unit_{task}_map{m}.npz")
        assert list(D["scen"]) == list(M["scen"])
        S = {t: (D if t in DIGIT_R else M)[f"S_{t}"] for t in READ}
        N = {t: (D if t in DIGIT_R else M)[f"N_{t}"] for t in READ}
        sh = assemble_shared(S, N, a.n_splits, SEED["split"] + zlib.crc32(task.encode()) % 100000)
        G = dict(zip(READ, sh["G"]))
        g = sh["g"]
        jd = json.load(open(out / f"unit_{task}_digit.json"))
        jm = json.load(open(out / f"unit_{task}_map{m}.json"))
        valid = jd["valid"] | jm["valid"]
        np.savez(fo, scen=D["scen"], **sh)
        (out / f"shared_{task}_map{m}.json").write_text(json.dumps({
            "task": task, "map": m, "valid": valid, "enter": task_entry(valid),
            "valid_judgment": [t for t in JUDG if valid[t]], "valid_control": [t for t in CTRL if valid[t]],
            "g_norm": float(np.linalg.norm(g)),
            "resid_norm_frac": float(np.linalg.norm(g) / np.mean([np.linalg.norm(G[t]) for t in JUDG]))}, indent=1))

    for task in tasks:
        jd = run_unit(task, "digit", DIGIT_R, "1", None)
        for m in maps:
            run_unit(task, f"map{m}", LABEL_R, m, jd["ref_J"])
            assemble(task, m)
        run.setdefault("done_tasks", []).append(task)
        save_run()

    # exploratory baselines C1-C4 (one readout each; no validity, no control span)
    for task in [t for t in a.baselines.split(",") if t]:
        fo = out / f"base_{task}.npz"
        if fo.exists():
            continue
        r = [x for x in R["1"] if x["ptask"] == task]
        if keep_scen is not None:
            r = r[: 2 * a.max_scen]
        W, ch = baseline_weights(r)
        ro = Readout(tok, r, ch, W, last_only=a.grad_pos == "last")
        gi = ro.grads(model, steer, hdim, bs=a.bs)
        np.savez(fo, g=gi.mean(0), n=len(r))

    # finite-difference check of the J gradient (P2 amendment 01 criterion: corr over random directions at alpha/4,
    # and the ratio along the unit gradient at alpha/16)
    if a.fd_check:
        task = tasks[0]
        items = canonical_items(rows("1", task, "J"), "J")
        W, ch = score_weights(items, "J")
        ro = Readout(tok, items, ch, W, last_only=a.grad_pos == "last")
        g = ro.grads(model, steer, hdim, bs=a.bs).mean(0)
        sc = lambda v_: float(((ro.logp(model, steer, v_)) * W).sum(1).mean())
        rng = np.random.default_rng(SEED["fd"])
        rows_ = []
        for _ in range(a.fd_check):
            u = unit(rng.standard_normal(hdim))
            e = alpha / 4
            rows_.append({"fd": (sc(e * u) - sc(-e * u)) / (2 * e), "grad": float(u @ g)})
        u = unit(g)
        e16 = alpha / 16
        r16 = (sc(e16 * u) - sc(-e16 * u)) / (2 * e16) / float(u @ g)
        fd, gr = [r["fd"] for r in rows_], [r["grad"] for r in rows_]
        run["fd_check"] = {"task": task, "rows": rows_, "g_ratio_alpha16": r16,
                           "corr_random": float(np.corrcoef(fd, gr)[0, 1]) if len(rows_) > 2 else None}
    run["t_total_s"] = time.time() - run["t0"]
    save_run()
    print(json.dumps({k: run.get(k) for k in ("layer", "alpha", "done_tasks", "fd_check", "t_total_s")},
                     default=float)[:3000])


if __name__ == "__main__":
    main()
