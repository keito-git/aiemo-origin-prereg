"""Stage 1i: B2m = B2 with both options worded identically except for the size of the sanction (e.g. "a fee of
200 dollars" / "a fee of 2 dollars"). This holds formality and normativity fixed (R3, Stage 1h review) and keeps the
valence gap small. Pairs chosen by the valence probe only (03_解析/stage1i_probe_mag/): small mean negativity gap over
the six planned models; no judgment readout was run when choosing. Target = the larger sanction.
Scenario, frame, option position and target are kept from B2 (same cyclic pair assignment)."""
import csv

MAGNITUDE = {  # B2 strict phrase -> (larger sanction, smaller sanction; same wording)
    " a fine": (" a fee of 200 dollars", " a fee of 2 dollars"),
    " a suspension": (" a suspension of ten days", " a suspension of ten minutes"),
    " a formal penalty": (" a full-day hearing", " a ten-minute hearing"),
    " a written reprimand": (" a month of probation", " a day of probation"),
    " a temporary ban": (" a two-month ban", " a two-hour ban"),
    " a disciplinary hearing": (" ten hours of extra duty", " ten minutes of extra duty"),
}
rows = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
out = []
for x in rows:
    strict = x["option_a"] if x["target_option"] == "a" else x["option_b"]
    s, l = MAGNITUDE[strict]
    a, b = (s, l) if x["target_option"] == "a" else (l, s)
    out.append({"item_id": x["item_id"].replace("B2_", "B2m_"), "readout": "B2m", "prompt": x["prompt"],
                "option_a": a, "option_b": b, "target_option": x["target_option"], "pair": strict.strip()})
with open("readout_B2m.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), sum(r["target_option"] == "a" for r in out))
print(out[0]); print(out[1])
