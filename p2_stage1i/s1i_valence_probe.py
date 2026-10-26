"""Stage 1i stimulus check (no steering, no judgment readout): negativity of candidate neutral option phrases,
log-odds of " negative" vs " positive" after 'The phrase "<option>" sounds'. Used only to choose option pairs whose
two options sound equally positive/negative."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_light_round import Steer, decoder_layers
from s1c_word_vs_situation import Readout

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--cands", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
pairs = json.load(open(a.cands))["pairs"]
phrases = sorted({p for pr in pairs for p in pr})
tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16, device_map="cuda").eval()
steer = Steer(decoder_layers(model)[0])
items = [{"prompt": f'The phrase "{p.strip()}" sounds', "option_a": " negative", "option_b": " positive",
          "target_option": "a"} for p in phrases]
neg = dict(zip(phrases, Readout(tok, items).score(model, steer, None).tolist()))
res = {"negativity": neg, "gap_lenient_minus_strict": {f"{s.strip()}|{l.strip()}": neg[l] - neg[s] for s, l in pairs}}
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res["gap_lenient_minus_strict"], indent=1))
