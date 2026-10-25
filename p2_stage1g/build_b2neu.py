"""Stage 1g amendment: B2neu = B2 with a stance-free frame, used only to estimate each item's option prior.
Target = the strict option (as in B2). Removing this prior from B2 and B2neg lets the manipulation check ask whether
the B2neg judgment tracks the B2 judgment, instead of being dominated by the shared option prior (which enters B2
and B2neg with opposite signs and drives their raw correlation negative even when the negation is understood)."""
import csv

FRAMES = {"The response chosen by the responsible official was": "The response mentioned in the report was",
          "According to the written policy, this case calls for": "According to the written policy, this case is about"}
rows = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
out = []
for x in rows:
    hits = [k for k in FRAMES if x["prompt"].endswith(k)]
    assert len(hits) == 1
    out.append({"item_id": x["item_id"].replace("B2_", "B2neu_"), "readout": "B2neu",
                "prompt": x["prompt"][: -len(hits[0])] + FRAMES[hits[0]],
                "option_a": x["option_a"], "option_b": x["option_b"], "target_option": x["target_option"]})
with open("readout_B2neu.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), sum(r["target_option"] == "a" for r in out))
print(out[0]); print(out[1])
