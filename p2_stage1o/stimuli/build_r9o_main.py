"""Stage 1o main readout stimuli (2026-11-03; design 11 §7; before any Stage 1o registration).
60 new rule-violation scenarios (L01..L60; 30 indoors, 30 outdoors, column loc as in Stage 1k), never used in earlier
stages. The Stage 1n scenarios (K01..K60) were used for the gradient decomposition and the exploratory analyses, so
Stage 1o reads out on new ones.
Every readout row is copied from the file the probes checked, with only the scenario sentence replaced
(K scenario i -> L scenario i); the severity sentence, the question, the options and the rotations are unchanged:
  S, HARSH, L, U_bright, U_fam, C   readout_R9n.csv
  CAT_S, CAT_U                      readout_R9o_catprobe.csv
  CAT_R                             readout_R9o_catprobe2.csv
  CAT_S2, PAIR                      readout_R9o_catprobe3.csv
Inputs are read only. Writes readout_R9o_main.csv."""
import csv
import re

# Stage 1k scenario sentences, read from readout_R9k.csv (build_r9k.py is not imported: it writes on import)
_MILD = " The person apologized right away and had a clean record.\n"
K_ALL = [r["prompt_qa"][len("Scenario: "):r["prompt_qa"].index(_MILD)]
         for r in csv.DictReader(open("readout_R9k.csv")) if r["task"] == "S" and r["sev"] == "mild"]
assert len(K_ALL) == 60
K_IN, K_OUT = K_ALL[:30], K_ALL[30:]

INDOOR = [
    "A cashier let a friend skip the queue at the post office counter.",
    "A tenant ran a loud washing machine in the apartment after the building's quiet hour.",
    "A library user tore a page out of a reference book in the reading room.",
    "An employee printed a stack of personal flyers on the office printer.",
    "A student used a classmate's access card to enter the dormitory.",
    "A waiter served a customer a dish without mentioning that it contained nuts.",
    "A gym member used the sauna without showering first, against the posted rule.",
    "A shop assistant sold a lottery ticket to a customer who looked underage.",
    "A cleaner mixed two cleaning products in the hotel bathroom despite the label warning.",
    "A visitor brought a dog into the hospital lobby where pets are not allowed.",
    "A student submitted an essay that reused large parts of an older essay of their own.",
    "An employee clocked in for a coworker who had not yet arrived at the office.",
    "A passenger put a bag on the seat of a full bus and refused to move it.",
    "A technician skipped a scheduled safety check on the elevator of an office building.",
    "A barista gave free drinks to friends at the cafe without recording them.",
    "A museum guard let a group in through the staff entrance without tickets.",
    "A homeowner's guest smoked in the shared hallway of the apartment building.",
    "A receptionist read a guest's mail at the hotel front desk.",
    "A trainee changed a figure in an office spreadsheet to hide a small error.",
    "A student ate a full meal at a computer in the school computer room against the rules.",
    "A daycare worker left the gate between two playrooms unlatched.",
    "A pharmacist sold a box of cold medicine above the daily limit to one customer.",
    "A store manager kept the emergency exit of the shop locked during opening hours.",
    "A hairdresser reused a comb on a new customer without cleaning it in the salon.",
    "A tutor gave a student the exam questions a day early in the study center.",
    "A bowling alley worker let customers play after closing time and kept the fee.",
    "A dental assistant looked up a neighbor's record on the clinic computer.",
    "A choir member recorded a closed rehearsal in the church hall and posted it online.",
    "An accountant sent a client's tax form to the wrong email address from the office.",
    "A cinema worker let friends watch a film for free from the back of the theater.",
]
OUTDOOR = [
    "A driver made a U-turn across a double solid line on a busy avenue.",
    "A dog owner let a dog run off the leash on a beach where leashes are required.",
    "A cyclist rode on the sidewalk of a crowded shopping street.",
    "A builder left a deep hole on a building site without a fence around it.",
    "A hunter shot at birds a week before the hunting season opened.",
    "A resident burned garden leaves in a backyard during a fire ban.",
    "A driver did not stop for a pedestrian at a marked crosswalk.",
    "A tourist fed bread to the ducks at a city pond despite a sign asking people not to.",
    "A landscaper dumped grass cuttings in a vacant lot next to the road.",
    "A runner cut across a private farm field to shorten the route.",
    "A scooter rider carried a passenger on a one-person rental scooter in the city center.",
    "A driver overtook a school bus that had stopped with its lights flashing.",
    "A climber bolted a new route on a cliff where bolting is banned.",
    "A shop owner placed tables on the sidewalk beyond the line marked on the permit.",
    "A visitor walked a bicycle across the lawns of the botanical garden after being told not to.",
    "A truck driver drove a heavy truck over a small bridge with a lower weight limit.",
    "A family left a portable barbecue burning on the grass of a public park.",
    "A kayaker paddled into a bird nesting zone of the lake marked as closed.",
    "A courier rode an electric bike on a forest path closed to motor vehicles.",
    "A resident let a garden hedge grow over the bicycle lane in front of the house.",
    "A camper washed dishes with soap directly in a mountain stream.",
    "A driver parked across the entrance of a neighbor's driveway overnight.",
    "A festival organizer kept the outdoor music going past the permitted hour in the square.",
    "A rider took a horse along a footpath reserved for walkers.",
    "A gardener sprayed weed killer near the edge of a public playground.",
    "A pilot of a small drone flew it over a crowd at an outdoor market.",
    "A fisherman kept a fish below the minimum size at the pier.",
    "A bus driver let passengers off in the middle of the road instead of at the stop.",
    "A teenager climbed the fence of the public swimming pool at night to swim.",
    "A van driver unloaded goods in a cycle lane on the high street.",
]

assert len(INDOOR) == len(OUTDOOR) == 30 and len(set(INDOOR + OUTDOOR)) == 60
assert not set(INDOOR + OUTDOOR) & set(K_IN + K_OUT)
OLD = K_IN + K_OUT
NEW = INDOOR + OUTDOOR
SRC = (("readout_R9n.csv", ("S", "HARSH", "L", "U_bright", "U_fam", "C")),
       ("readout_R9o_catprobe.csv", ("CAT_S", "CAT_U")),
       ("readout_R9o_catprobe2.csv", ("CAT_R",)),
       ("readout_R9o_catprobe3.csv", ("CAT_S2", "PAIR")))
fields = ["item_id", "task", "sev", "scen", "loc", "prompt_qa", "rot", "ranks"]
out = []
for fn, tasks in SRC:
    for r in csv.DictReader(open(fn)):
        if r["task"] not in tasks:
            continue
        i = int(r["scen"][1:]) - 1
        head = f"Scenario: {OLD[i]} "
        assert r["prompt_qa"].startswith(head), r["item_id"]
        assert int(r["loc"]) == (0 if i < 30 else 1)
        new = {k: r.get(k, "") for k in fields}
        new["scen"] = f"L{i + 1:02d}"
        new["item_id"] = re.sub(r"K\d\d$", new["scen"], r["item_id"])
        assert new["item_id"] != r["item_id"], r["item_id"]
        new["item_id"] = new["item_id"].replace("R9K_", "R9OM_").replace("R9O_", "R9OM_")
        new["prompt_qa"] = f"Scenario: {NEW[i]} " + r["prompt_qa"][len(head):]
        out.append(new)
n = {}
for r in out:
    n[(r["task"], r["rot"])] = n.get((r["task"], r["rot"]), 0) + 1
assert all(v == 120 for v in n.values()), n
assert len({r["item_id"] for r in out}) == len(out)
with open("readout_R9o_main.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)
print("rows", len(out), "blocks", len(n))
print(out[0]["prompt_qa"])
print([r for r in out if r["task"] == "PAIR"][-1]["prompt_qa"])
