"""Stage 1j (after the format probe): forward-scale 1-9 ratings with a built-in severity manipulation.
The format probe (s1j_format_probe.py) showed that none of the six models follows swapped scale anchors, so only the
forward scale is used (1 = no sanction at all / extremely light, 9 = the most severe sanction / extremely heavy).
Severity manipulation (validity check of any readout): each of the 50 B2 scenarios gets one added sentence,
  mild:   "The person apologized right away and had a clean record."
  severe: "The person had done this many times before, despite several written warnings."
Outputs:
  readout_R9s.csv: task S = 50 scenarios x {mild, severe} x 2 frames (frame type as in B2), task O = 50 objects x 2
                   frames (from readout_R9.csv, scale f). Columns: item_id, task, sev, frame, scen, prompt_cont,
                   prompt_qa, weight_kg.
  readout_B2sev.csv: the 100 B2 items with the same sentence inserted before the frame (200 rows), same options.
"""
import csv

FRAMES_B2 = (" The response chosen by the responsible official was",
             " According to the written policy, this case calls for")
SEV = {"mild": "The person apologized right away and had a clean record.",
       "severe": "The person had done this many times before, despite several written warnings."}
S_CONT = ("{s} The responsible official rated the sanction this case deserves on a scale from 1 (no sanction at all) "
          "to 9 (the most severe sanction). The rating was",
          "{s} According to the written policy, the sanction this case calls for is rated on a scale from 1 (no "
          "sanction at all) to 9 (the most severe sanction). The rating is")
S_QA = "Scenario: {s}\nQuestion: On a scale from 1 (no sanction at all) to 9 (the most severe sanction), how severe a " \
       "sanction does this case deserve? Answer with one digit."
O_QA = "Object: {o}\nQuestion: On a scale from 1 (extremely light) to 9 (extremely heavy), how heavy is it? Answer " \
       "with one digit."

b2 = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
r9 = [x for x in csv.DictReader(open("readout_R9.csv")) if x["task"] == "O" and x["scale"] == "f"]
out, b2s = [], []
for x in b2:
    k = [i for i, f in enumerate(FRAMES_B2) if x["prompt"].endswith(f)]
    assert len(k) == 1
    scen = x["prompt"][: -len(FRAMES_B2[k[0]])]
    for sev, sent in SEV.items():
        s = f"{scen} {sent}"
        n = x["item_id"].split("_")[1]
        out.append({"item_id": f"R9S_{sev}_{n}", "task": "S", "sev": sev, "frame": k[0], "scen": n,
                    "prompt_cont": S_CONT[k[0]].format(s=s), "prompt_qa": S_QA.format(s=s), "weight_kg": ""})
        b2s.append(x | {"item_id": f"B2sev_{sev}_{n}", "readout": "B2sev", "prompt": s + FRAMES_B2[k[0]]})
for x in r9:
    o = x["prompt"].split(".")[0].replace("Consider ", "")
    out.append({"item_id": x["item_id"].replace("R9Of_", "R9O_"), "task": "O", "sev": "", "frame": x["frame"],
                "scen": "", "prompt_cont": x["prompt"], "prompt_qa": O_QA.format(o=o), "weight_kg": x["weight_kg"]})
for name, rows in (("readout_R9s.csv", out), ("readout_B2sev.csv", b2s)):
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print(len(out), len(b2s)); print(out[1]); print(out[-1]); print(b2s[1])
