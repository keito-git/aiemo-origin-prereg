"""Stage 1m stimuli (2026-10-29).
1. stage1m_closings.json: neu_hi / neu_lo, a non-emotional intensity contrast (size, height, weight, amount, weather).
   Sentence k mirrors anger_L[k] / anger_lo[k] (Stage 1d / 1l) in form where possible (same intensifier slot: "so",
   "absolutely", "really", "extremely" vs "slightly", "mildly", "fairly") and has the word count of neutral_C[k].
2. readout_R9m.csv: readout_R9k.csv (unchanged rows) plus task L = the same scenario text with a leniency question,
   where a higher digit means a milder response. A harsher judgment lowers L; a bias towards "more of whatever is
   asked" raises it.
Inputs are read only. Writes the two new files."""
import csv
import json

NEU = {
    "neu_hi": ["It was enormous.", "The street was so wide.", "It was absolutely huge.",
               "I noticed the temperature soaring.", "The bag was extremely heavy too.", "The hall was gigantic.",
               "The queue was enormous the whole evening.", "It was extremely windy all day.",
               "It was immense in size.", "The building was extremely tall there.", "The pile was gigantic.",
               "It filled the entire hall."],
    "neu_lo": ["It was large.", "The street was fairly wide.", "It was fairly big.",
               "I noticed the temperature rising.", "The bag was slightly heavy too.", "The hall was big.",
               "The queue was long the whole evening.", "It was slightly windy all day.",
               "It was moderate in size.", "The building was fairly tall there.", "The pile was sizeable.",
               "It filled half the hall."],
}
ref = json.load(open("stage1d_closings.json"))["neutral_C"]
for k, v in NEU.items():
    assert len(v) == 12 and len(set(v)) == 12, k
    for i, s in enumerate(v):
        assert len(s.split()) == len(ref[i].split()), (k, i, s, ref[i])
json.dump({"note": __doc__.split("2. readout")[0].strip()} | NEU, open("stage1m_closings.json", "w"), indent=1)

Q_S = ("On a scale from 1 (no sanction at all) to 9 (the most severe sanction), how severe a sanction does this case "
       "deserve? Answer with one digit.")
Q_L = ("On a scale from 1 (not lenient at all) to 9 (extremely lenient), how leniently should this case be treated? "
       "Answer with one digit.")
rows = list(csv.DictReader(open("readout_R9k.csv")))
new = []
for r in rows:
    if r["task"] != "S":
        continue
    assert r["prompt_qa"].endswith("Question: " + Q_S), r["item_id"]
    new.append(r | {"item_id": r["item_id"].replace("R9K_S_", "R9M_L_"), "task": "L",
                    "prompt_qa": r["prompt_qa"].replace(Q_S, Q_L)})
assert len(new) == 120
with open("readout_R9m.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows + new)
print({k: len(v) for k, v in NEU.items()}, "R9m rows", len(rows) + len(new))
