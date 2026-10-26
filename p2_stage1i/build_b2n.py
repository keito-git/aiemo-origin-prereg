"""Stage 1i: B2n = B2 with both options worded neutrally; they differ in formality/severity but sound about equally
positive/negative. Pairs chosen from neutral_pair_candidates.json by the valence probe (s1i_valence_probe.py,
results in 03_解析/stage1i_probe/): smallest mean |negativity gap| over the six planned models, signs mixed so the
mean gap over pairs is near zero. No judgment readout was run when choosing. Target = the stricter (formal) option.
Scenario, frame, option position and target are kept from B2 (same cyclic pair assignment)."""
import csv

NEUTRAL = {  # B2 strict phrase -> (neutral strict, neutral lenient)
    " a fine": (" a formal response", " an informal response"),
    " a suspension": (" a written notice", " a spoken note"),
    " a formal penalty": (" a hearing", " a chat"),
    " a written reprimand": (" a written record", " a verbal mention"),
    " a temporary ban": (" a formal process", " a simple mention"),
    " a disciplinary hearing": (" a formal letter", " a short email"),
}
rows = [x for x in csv.DictReader(open("readout_B.csv")) if x["readout"] == "B2"]
out = []
for x in rows:
    strict = x["option_a"] if x["target_option"] == "a" else x["option_b"]
    s, l = NEUTRAL[strict]
    a, b = (s, l) if x["target_option"] == "a" else (l, s)
    out.append({"item_id": x["item_id"].replace("B2_", "B2n_"), "readout": "B2n", "prompt": x["prompt"],
                "option_a": a, "option_b": b, "target_option": x["target_option"], "pair": strict.strip()})
with open("readout_B2n.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), sum(r["target_option"] == "a" for r in out))
print(out[0]); print(out[1])
