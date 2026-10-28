"""Diagnostic: on the RTX PRO 6000 (aiemo5), does the default sdpa attention in bf16 give the same digit probabilities
as eager attention, and is it deterministic? (fp32 sdpa was found wrong for >= ~25 tokens on another container.)
Compares baseline and steered (one fixed random direction) R9k digit probabilities: sdpa run 1, sdpa run 2, eager."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_light_round import Steer, decoder_layers, last_hidden, load_csv, unit  # noqa: E402
from s1j_rating import Rating  # noqa: E402

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--stim", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = load_csv(Path(a.stim) / "readout_R9k.csv")
for x in R:
    x["prompt"] = x["prompt_qa"] + "\nAnswer:"
cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
L = int(round(0.33 * cfg.num_hidden_layers))
res, P = {"model": a.model, "n_tokens": [len(tok(x["prompt"])["input_ids"]) for x in R[:5]]}, {}
v = None
for impl in ("sdpa", "eager"):
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation=impl, low_cpu_mem_usage=True).eval()
    steer = Steer(decoder_layers(model)[L - 1])
    Ra = Rating(tok, R)
    if v is None:
        H = last_hidden(model, tok, [x["prompt"] for x in R[:50]], L)
        alpha = 0.2 * float(np.linalg.norm(H, axis=1).mean())
        v = alpha * unit(np.random.default_rng(0).standard_normal(H.shape[1]))
    for rep in ((1, 2) if impl == "sdpa" else (1,)):
        P[f"{impl}{rep}_base"] = Ra.probs(model, steer, None)
        P[f"{impl}{rep}_steer"] = Ra.probs(model, steer, v)
    del model
    torch.cuda.empty_cache()
K = np.arange(1, 10)
for c in ("base", "steer"):
    for a1, a2 in (("sdpa1", "sdpa2"), ("sdpa1", "eager1")):
        x, y = P[f"{a1}_{c}"], P[f"{a2}_{c}"]
        res[f"{c}:{a1}_vs_{a2}"] = {"max_abs_dp": float(np.abs(x - y).max()), "max_abs_dE": float(np.abs((x - y) @ K).max()),
                                    "corr_E": float(np.corrcoef(x @ K, y @ K)[0, 1])}
d_s = (P["sdpa1_steer"] - P["sdpa1_base"]) @ K
d_e = (P["eager1_steer"] - P["eager1_base"]) @ K
res["steer_effect_corr_sdpa_eager"] = float(np.corrcoef(d_s, d_e)[0, 1])
res["steer_effect_mean"] = {"sdpa": float(d_s.mean()), "eager": float(d_e.mean())}
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / f"diag_sdpa_bf16_{a.model.split('/')[-1]}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
