"""Stage 1g: B2neg = B2 with the frame negated, so that choosing the strict option now means a lenient stance.
Target = the lenient option (the option a strict judge would name after "rejected" / "does not call for").
If steering changes the judgment, Delta(B2neg) has the same sign as Delta(B2); if it only raises the probability
of strict-sanction tokens, Delta(B2neg) is negative. Option order is kept, so positions stay counterbalanced."""
import csv

FRAMES = {"The response chosen by the responsible official was": "The response rejected by the responsible official was",
          "According to the written policy, this case calls for": "According to the written policy, this case does not call for"}
rows = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
out = []
for x in rows:
    hits = [k for k in FRAMES if x["prompt"].endswith(k)]
    assert len(hits) == 1
    out.append({"item_id": x["item_id"].replace("B2_", "B2neg_"), "readout": "B2neg",
                "prompt": x["prompt"][: -len(hits[0])] + FRAMES[hits[0]],
                "option_a": x["option_a"], "option_b": x["option_b"],
                "target_option": "b" if x["target_option"] == "a" else "a"})
with open("readout_B2neg.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), sum(r["target_option"] == "a" for r in out))
print(out[0]); print(out[1])
