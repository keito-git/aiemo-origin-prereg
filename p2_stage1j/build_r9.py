"""Stage 1j: R9 = 1-9 rating readout with no option words.
S (sanction): the 100 B2 items (same scenario and frame type, aligned one-to-one with B2 by b2_item); the model rates
  the deserved sanction on a 1-9 scale. Scale 'f': 1 = no sanction at all, 9 = the most severe sanction;
  scale 'r': anchors swapped. Strictness score = E[digit] for f and 10 - E[digit] for r.
O (object weight, control unrelated to the scenarios): 50 everyday objects x 2 frames; the model rates weight,
  1 = extremely light, 9 = extremely heavy (f) or swapped (r). weight_kg is a rough typical mass, used only for the
  rank check of the readout (Spearman with the model's baseline rating).
"""
import csv

FRAMES_B2 = (" The response chosen by the responsible official was",
             " According to the written policy, this case calls for")
S_TEMPL = ("{s} The responsible official rated the sanction this case deserves on a scale from 1 ({lo}) to 9 ({hi}). "
           "The rating was",
           "{s} According to the written policy, the sanction this case calls for is rated on a scale from 1 ({lo}) "
           "to 9 ({hi}). The rating is")
S_ANCH = ("no sanction at all", "the most severe sanction")
O_TEMPL = ("Consider {s}. An expert rated its weight on a scale from 1 ({lo}) to 9 ({hi}). The rating was",
           "Consider {s}. According to a reference table, its weight is rated on a scale from 1 ({lo}) to 9 ({hi}). "
           "The rating is")
O_ANCH = ("extremely light", "extremely heavy")
OBJECTS = [  # (object, rough typical mass in kg)
    ("a snowflake", 3e-6), ("a grain of rice", 3e-5), ("a postage stamp", 6e-5), ("a paperclip", 1e-3),
    ("a sheet of paper", 5e-3), ("a coin", 6e-3), ("a pencil", 7e-3), ("a house key", 1e-2), ("a golf ball", 0.046),
    ("a tennis ball", 0.058), ("an egg", 0.06), ("a deck of cards", 0.1), ("a banana", 0.12), ("a baseball", 0.145),
    ("an apple", 0.15), ("a smartphone", 0.18), ("a coffee mug", 0.3), ("a paperback book", 0.3),
    ("a soccer ball", 0.43), ("a loaf of bread", 0.5), ("a one-liter bottle of water", 1.0), ("a pineapple", 1.5),
    ("a laptop", 1.8), ("a brick", 2.5), ("a house cat", 4.5), ("a bowling ball", 6.5), ("a car tire", 10),
    ("a microwave oven", 15), ("a packed suitcase", 20), ("a bag of cement", 25), ("an office desk", 40),
    ("an adult sheep", 70), ("an adult man", 75), ("a washing machine", 75), ("a refrigerator", 90),
    ("a motorcycle", 200), ("a grand piano", 400), ("a horse", 500), ("a cow", 700), ("a small sailboat", 1000),
    ("a car", 1500), ("a pickup truck", 2500), ("an elephant", 5000), ("a school bus", 11000),
    ("a loaded truck", 30000), ("a train locomotive", 100000), ("a blue whale", 150000), ("a jet airliner", 300000),
    ("a cargo ship", 5e7), ("an aircraft carrier", 1e8),
]
assert len(OBJECTS) == 50

rows = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
out = []
for x in rows:
    k = [i for i, f in enumerate(FRAMES_B2) if x["prompt"].endswith(f)]
    assert len(k) == 1, x["prompt"]
    scen = x["prompt"][: -len(FRAMES_B2[k[0]])]
    for sc, (lo, hi) in (("f", S_ANCH), ("r", S_ANCH[::-1])):
        out.append({"item_id": x["item_id"].replace("B2_", f"R9S{sc}_"), "task": "S", "scale": sc, "frame": k[0],
                    "prompt": S_TEMPL[k[0]].format(s=scen, lo=lo, hi=hi), "b2_item": x["item_id"], "weight_kg": ""})
for i, (o, w) in enumerate(OBJECTS):
    for fr in (0, 1):
        for sc, (lo, hi) in (("f", O_ANCH), ("r", O_ANCH[::-1])):
            out.append({"item_id": f"R9O{sc}_{2 * i + fr + 1:03d}", "task": "O", "scale": sc, "frame": fr,
                        "prompt": O_TEMPL[fr].format(s=o, lo=lo, hi=hi), "b2_item": "", "weight_kg": w})
with open("readout_R9.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), {(t, s): sum(r["task"] == t and r["scale"] == s for r in out) for t in "SO" for s in "fr"})
for r in (out[0], out[1], out[2], out[-1]):
    print(r)
