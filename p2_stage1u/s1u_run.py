"""Stage 1u: run the registered Stage 1o / 1p code unchanged, with the fixed F8 format prefix on every readout prompt.
Usage: s1u_run.py o <s1o_rating.py args> | s1u_run.py p <s1p_rating.py args> | s1u_run.py check --model M --revision R
Two patches, nothing else:
  1. load_csv of readout_R9o_main.csv returns prompt_qa preceded by prefix(8) of s1u_prefix_pilot.py
     (the registered code then appends "\nAnswer:" as before). Manipulation checks and direction texts are unchanged.
  2. Readout.logp reuses the key/value cache of the prefix that all items of a readout share (the longest common token
     prefix). The steering vector is added at every position, prefix included, as before; the cache is rebuilt for
     every vector. Readout.grads is unchanged (full sequences).
'check' compares the cached and the full logp on every readout, without and with one random vector.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
import torch

import s1_light_round
import s1o_rating
from s1u_prefix_pilot import prefix

PREFIX = prefix(8)
_load_csv = s1_light_round.load_csv


def load_csv_prefixed(path):
    rows = _load_csv(path)
    if str(path).endswith("readout_R9o_main.csv"):
        for x in rows:
            x["prompt_qa"] = PREFIX + x["prompt_qa"]
    return rows


def common_len(seqs) -> int:
    n = min(len(s) for s in seqs) - 1          # keep at least one token per item outside the cache
    first = seqs[0]
    for s in seqs[1:]:
        k = 0
        while k < n and s[k] == first[k]:
            k += 1
        n = k
    return n


_full_logp = s1o_rating.Readout.logp


@torch.no_grad()
def cached_logp(self, model, steer, v, bs: int = 64) -> np.ndarray:
    if not hasattr(self, "n_pre"):
        self.n_pre = common_len(self.seqs)
    P = self.n_pre
    if P < 8:
        return _full_logp(self, model, steer, v, bs)
    steer.v = None if v is None else torch.tensor(v, device="cuda")
    pre = torch.tensor(self.seqs[0][:P], device="cuda")[None]
    pkv = model(input_ids=pre, attention_mask=torch.ones_like(pre), use_cache=True).past_key_values
    out = []
    for b in range(0, len(self.seqs), bs):
        s = [x[P:] for x in self.seqs[b:b + bs]]
        T = max(len(x) for x in s)
        ids = torch.full((len(s), T), self.pad, dtype=torch.long)
        mask = torch.zeros((len(s), P + T), dtype=torch.long)
        mask[:, :P] = 1
        for j, x in enumerate(s):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, P:P + len(x)] = 1
        cache = copy.deepcopy(pkv)
        cache.batch_repeat_interleave(len(s))
        last = torch.tensor([len(x) - 1 for x in s], device="cuda")
        lg = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), past_key_values=cache, use_cache=True).logits
        out.append(torch.log_softmax(lg[torch.arange(len(s)), last][:, self.choice].float(), -1).cpu().numpy())
        del cache, lg
    steer.v = None
    R = np.empty((len(self.seqs), len(self.choice)))
    R[self.order] = np.concatenate(out)
    return R


def patch():
    s1o_rating.load_csv = load_csv_prefixed
    s1o_rating.Readout.logp = cached_logp
    import s1p_rating
    s1p_rating.load_csv = load_csv_prefixed


def check(argv):
    import argparse
    import json
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from s1p_rating import build_readouts
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--stim", required=True)
    a = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True).eval()
    steer = s1_light_round.Steer(s1_light_round.decoder_layers(model)[int(round(0.33 * cfg.num_hidden_layers)) - 1])
    R9 = load_csv_prefixed(f"{a.stim}/readout_R9o_main.csv")
    for x in R9:
        x["prompt"] = x["prompt_qa"] + "\nAnswer:"
    RO, _ = build_readouts(tok, R9)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(cfg.hidden_size)
    v = v / np.linalg.norm(v) * 8.0
    res = {}
    for t, r in RO.items():
        res[t] = {"n_pre": common_len(r.seqs), "len_max": max(len(x) for x in r.seqs)}
        for nm, vec in (("none", None), ("rand", v)):
            f, c = _full_logp(r, model, steer, vec), cached_logp(r, model, steer, vec)
            res[t][nm] = {"max_abs_logp": float(np.abs(f - c).max()),
                          "max_abs_score": float(np.abs(((f - c) * r.W).sum(1)).max())}
        import time
        t0 = time.time()
        f0, fv = _full_logp(r, model, steer, None), _full_logp(r, model, steer, v)
        t1 = time.time()
        c0, cv = cached_logp(r, model, steer, None), cached_logp(r, model, steer, v)
        t2 = time.time()
        g0, gv = _full_logp(r, model, steer, None, bs=16), _full_logp(r, model, steer, v, bs=16)
        ef, ec = ((fv - f0) * r.W).sum(1).mean(), ((cv - c0) * r.W).sum(1).mean()
        eg = ((gv - g0) * r.W).sum(1).mean()
        # reference noise of the registered (full) computation: the same effect with batch size 16 instead of 64
        res[t].update({"effect_full": float(ef), "effect_cached": float(ec), "effect_full_bs16": float(eg),
                       "diff_cached": float(ec - ef), "diff_full_bs16": float(eg - ef),
                       "sec_full": t1 - t0, "sec_cached": t2 - t1})
        print(t, json.dumps(res[t]), flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "check":
        check(rest)
    else:
        patch()
        mod = s1o_rating if mode == "o" else __import__("s1p_rating")
        sys.argv = [f"s1{mode}_rating.py"] + rest
        mod.main()
