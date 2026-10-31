"""Stage 1n readout probe stimuli (2026-10-31, before any Stage 1n registration).
readout_R9n_probe.csv: the Stage 1k scenario texts (60 x mild/severe) with candidate questions, all 1-9 with the
high end phrased as "extremely ..." or "the most ...":
  judgment, harsh-high : S (Stage 1k), HARSH
  judgment, lenient-high: L (Stage 1m), MILD
  judgment-unrelated degree questions about the place (candidates for a degree control U): U_crowd, U_noise, U_old,
  U_fam
  C (Stage 1k, indoors/outdoors)
A degree-projection account (R3 C') predicts that an intensity direction raises every question; a harsher judgment
predicts S, HARSH up, L, MILD down and U unchanged. The probe only checks the baseline readouts (no steering).
Probe 2 (same day, after the design review R3): further U candidates that differ in how close their wording is to
"intensity" (U_size close to the int_neu pairs, U_bright, U_formal, U_unusual about the event). Written to
readout_R9n_probe2.csv so that the first probe file stays unchanged.
Inputs are read only. Writes readout_R9n_probe.csv and readout_R9n_probe2.csv."""
import csv

Q_S = ("On a scale from 1 (no sanction at all) to 9 (the most severe sanction), how severe a sanction does this case "
       "deserve? Answer with one digit.")
Q = {
    "L": ("On a scale from 1 (not lenient at all) to 9 (extremely lenient), how leniently should this case be "
          "treated? Answer with one digit."),
    "HARSH": ("On a scale from 1 (not harshly at all) to 9 (extremely harshly), how harshly should this case be "
              "treated? Answer with one digit."),
    "MILD": ("On a scale from 1 (not mild at all) to 9 (extremely mild), how mild a response does this case call "
             "for? Answer with one digit."),
    "U_crowd": ("On a scale from 1 (not crowded at all) to 9 (extremely crowded), how crowded was the place where "
                "this happened? Answer with one digit."),
    "U_noise": ("On a scale from 1 (completely quiet) to 9 (extremely noisy), how noisy was the place where this "
                "happened? Answer with one digit."),
    "U_old": ("On a scale from 1 (not old at all) to 9 (extremely old), how old is the place where this happened? "
              "Answer with one digit."),
    "U_fam": ("On a scale from 1 (not familiar at all) to 9 (extremely familiar), how familiar is this kind of place "
              "to most people? Answer with one digit."),
}
rows = list(csv.DictReader(open("readout_R9k.csv")))
new = []
for r in rows:
    if r["task"] != "S":
        continue
    assert r["prompt_qa"].endswith("Question: " + Q_S), r["item_id"]
    for t, q in Q.items():
        new.append(r | {"item_id": r["item_id"].replace("R9K_S_", f"R9N_{t}_"), "task": t,
                        "prompt_qa": r["prompt_qa"].replace(Q_S, q)})
assert len(new) == 120 * len(Q)
with open("readout_R9n_probe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows + new)
print("R9n_probe rows", len(rows) + len(new))

Q2 = {
    "U_size": ("On a scale from 1 (very small) to 9 (extremely large), how large was the place where this happened? "
               "Answer with one digit."),
    "U_bright": ("On a scale from 1 (very dim) to 9 (extremely bright), how brightly lit was the place where this "
                 "happened? Answer with one digit."),
    "U_formal": ("On a scale from 1 (not formal at all) to 9 (extremely formal), how formal is the place where this "
                 "happened? Answer with one digit."),
    "U_unusual": ("On a scale from 1 (not unusual at all) to 9 (extremely unusual), how unusual is the place where "
                  "this happened? Answer with one digit."),
}
new2 = []
for r in rows:
    if r["task"] != "S":
        continue
    for t, q in Q2.items():
        new2.append(r | {"item_id": r["item_id"].replace("R9K_S_", f"R9N_{t}_"), "task": t,
                         "prompt_qa": r["prompt_qa"].replace(Q_S, q)})
assert len(new2) == 120 * len(Q2)
with open("readout_R9n_probe2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows([r for r in rows if r["task"] == "S"] + new2)
print("R9n_probe2 rows", 120 + len(new2))
