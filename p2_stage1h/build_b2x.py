"""Stage 1h: B2x = B2 with option wording crossed against severity. The strict option is worded in neutral or
positive terms and the lenient option in negative terms. Target = the strict option (as in B2).
If steering makes the judgment stricter, Delta(B2x) has the same sign as Delta(B2). If it only raises the probability
of negative-sounding tokens, Delta(B2x) is negative. Scenario, frame, option position and target are kept from B2;
each B2 pair is replaced by one crossed pair (same cyclic assignment)."""
import csv

CROSSED = {  # B2 strict phrase -> (crossed strict, crossed lenient)
    " a fine": (" full accountability", " careless neglect"),
    " a suspension": (" a firm and fair response", " ignoring the problem"),
    " a formal penalty": (" upholding the rules in full", " looking the other way"),
    " a written reprimand": (" proper enforcement of the rules", " turning a blind eye"),
    " a temporary ban": (" responsible enforcement", " lax indifference"),
    " a disciplinary hearing": (" a just and formal response", " sloppy indifference"),
}
rows = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
out = []
for x in rows:
    strict = x["option_a"] if x["target_option"] == "a" else x["option_b"]
    s, l = CROSSED[strict]
    a, b = (s, l) if x["target_option"] == "a" else (l, s)
    out.append({"item_id": x["item_id"].replace("B2_", "B2x_"), "readout": "B2x", "prompt": x["prompt"],
                "option_a": a, "option_b": b, "target_option": x["target_option"], "pair": strict.strip()})
with open("readout_B2x.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), sum(r["target_option"] == "a" for r in out))
print(out[0]); print(out[1])
