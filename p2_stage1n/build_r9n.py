"""Stage 1n readout stimuli (2026-10-31, before any steered Stage 1n run).
readout_R9n.csv: the 60 Stage 1k scenarios x mild/severe with seven questions, all 1-9:
  S, C, L          the readout_R9m.csv rows, unchanged
  HARSH, U_fam     the readout_R9n_probe.csv rows, unchanged
  U_bright, U_size the readout_R9n_probe2.csv rows, unchanged
The rows are copied from the probe files, so the question wording is exactly the one the probes checked.
Inputs are read only. Writes readout_R9n.csv."""
import csv

TASKS = ("S", "C", "L", "HARSH", "U_bright", "U_fam", "U_size")
SRC = {"S": "readout_R9m.csv", "C": "readout_R9m.csv", "L": "readout_R9m.csv", "HARSH": "readout_R9n_probe.csv",
       "U_fam": "readout_R9n_probe.csv", "U_bright": "readout_R9n_probe2.csv", "U_size": "readout_R9n_probe2.csv"}
cache = {f: list(csv.DictReader(open(f))) for f in set(SRC.values())}
fields = list(cache["readout_R9m.csv"][0])
assert all(list(r[0]) == fields for r in cache.values())
out = []
for t in TASKS:
    rows = [r for r in cache[SRC[t]] if r["task"] == t]
    assert len(rows) == 120, (t, len(rows))
    out += rows
# same scenario order in every task and severity block
for s in ("mild", "severe"):
    ref = [r["scen"] for r in out if r["task"] == "S" and r["sev"] == s]
    assert all([r["scen"] for r in out if r["task"] == t and r["sev"] == s] == ref for t in TASKS)
# the S rows of the three source files are identical
for f in ("readout_R9n_probe.csv", "readout_R9n_probe2.csv"):
    assert [r for r in cache[f] if r["task"] == "S"] == [r for r in cache["readout_R9m.csv"] if r["task"] == "S"]
assert len({r["item_id"] for r in out}) == len(out)
with open("readout_R9n.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)
print("R9n rows", len(out))
