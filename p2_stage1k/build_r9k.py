"""Stage 1k stimuli: 60 new rule-violation scenarios (never used in earlier stages), each rated on two questions about
the same text.
  S: deserved sanction, 1 (no sanction at all) - 9 (the most severe sanction)   (as in Stage 1j)
  C: location, 1 (certainly indoors) - 9 (certainly outdoors)                   (control on the same text)
Location is designed: 30 scenarios take place indoors and 30 outdoors (column loc), so the control has its own
validity check (Spearman of the C rating with loc). Each scenario gets the Stage 1j mild / severe added sentence, which
should move S but not C.
Output readout_R9k.csv: item_id, task (S/C), sev (mild/severe), scen (K01..K60), loc (0 indoor / 1 outdoor), prompt_qa.
"""
import csv

SEV = {"mild": "The person apologized right away and had a clean record.",
       "severe": "The person had done this many times before, despite several written warnings."}
Q = {"S": "On a scale from 1 (no sanction at all) to 9 (the most severe sanction), how severe a sanction does this "
          "case deserve? Answer with one digit.",
     "C": "On a scale from 1 (certainly indoors) to 9 (certainly outdoors), where did this take place? Answer with "
          "one digit."}
INDOOR = [
    "An employee used the company courier account to send personal parcels from the office.",
    "A student talked loudly on the phone in the quiet section of the library.",
    "A shopper opened a packaged snack inside the supermarket and ate it before paying.",
    "A worker left a box of supplies blocking the hallway of the clinic.",
    "A student used the school chemistry lab equipment after hours without supervision.",
    "A passenger played music without headphones inside the train car.",
    "A gym member left heavy weights on the floor of the gym instead of returning them to the rack.",
    "A student copied two homework answers from a classmate in the classroom.",
    "An employee took several pens and notebooks from the office supply cabinet for home use.",
    "A visitor touched a fragile sculpture inside the museum gallery despite the signs.",
    "A restaurant worker skipped the required hand-washing step in the kitchen.",
    "A hotel guest smoked a cigarette inside a non-smoking room.",
    "An employee shared a login password with a coworker at the office.",
    "A cinema customer recorded part of the film on a phone inside the theater.",
    "A nurse left a medication cart unlocked in the hospital corridor.",
    "A warehouse worker stacked boxes higher than the marked limit inside the storage room.",
    "An office worker ate a coworker's labeled lunch from the shared kitchen refrigerator.",
    "A student brought a phone into the exam room against the rules.",
    "A store clerk gave a friend an unauthorized discount at the checkout counter.",
    "A lab assistant stored food in the laboratory refrigerator meant for samples.",
    "A resident propped open the fire door in the apartment stairwell.",
    "A call center agent ended a customer call early without resolving the issue.",
    "A patient used a phone to film other patients in the hospital waiting room.",
    "A factory worker removed a safety guard from a machine on the factory floor.",
    "An intern opened a confidential file on the office computer without permission.",
    "A bank teller left the cash drawer open while stepping away from the counter.",
    "A teacher left a class unsupervised in the classroom for twenty minutes.",
    "A cook reused frying oil past its replacement date in the restaurant kitchen.",
    "A pharmacy worker filled a prescription inside the pharmacy without checking the patient's identification.",
    "A guest at a conference center took extra food from the indoor buffet to bring home.",
]
OUTDOOR = [
    "A driver parked in a disabled parking space in the outdoor lot of a supermarket.",
    "A truck driver blocked a fire hydrant on the street for most of the afternoon.",
    "A resident left bags of trash next to the public bins in the park.",
    "A hiker walked off the marked trail into a protected area of the forest.",
    "A parent drove through the school parking lot faster than the posted limit.",
    "A driver crossed a red light at an empty intersection.",
    "A camper lit a small fire outside the designated fire pit at the campsite.",
    "A delivery driver left a van idling in the loading zone of the street for an hour.",
    "A student skateboarded on the school's outdoor stairs despite a posted ban.",
    "A resident watered the garden during a citywide water restriction.",
    "A fisherman caught fish in a section of the river closed for the season.",
    "A visitor climbed over a fence at an outdoor viewpoint to take photographs.",
    "A jogger dropped an empty bottle along the riverside path.",
    "A driver used a phone while driving on the highway.",
    "A neighbor set off fireworks in the backyard after the permitted hour.",
    "A worker on a construction site did not wear a hard hat.",
    "A park visitor picked flowers from the public garden beds.",
    "A motorcyclist rode without a helmet on a country road.",
    "A street vendor set up a stall on the sidewalk without a permit.",
    "A resident parked a car on the lawn of the shared courtyard.",
    "A boater sped through the marked slow zone of the harbor.",
    "A golfer drove a cart onto the green of the golf course.",
    "A tourist carved initials into a tree in the city park.",
    "A farmer burned crop waste in an open field without a permit.",
    "A taxi driver stopped in the bus lane on the main road to pick up a passenger.",
    "A soccer coach held practice on the field after it had been closed for repairs.",
    "A cyclist rode through the pedestrian zone of the town square.",
    "A student climbed onto the roof of the outdoor shelter in the schoolyard.",
    "A resident put up a large sign on a public lamppost without permission.",
    "A swimmer swam beyond the buoys at the public beach.",
]

assert len(INDOOR) == len(OUTDOOR) == 30 and len(set(INDOOR + OUTDOOR)) == 60
scen = [(s, 0) for s in INDOOR] + [(s, 1) for s in OUTDOOR]
rows = []
for task in ("S", "C"):
    for sev, sent in SEV.items():
        for i, (s, loc) in enumerate(scen):
            n = f"K{i + 1:02d}"
            rows.append({"item_id": f"R9K_{task}_{sev}_{n}", "task": task, "sev": sev, "scen": n, "loc": loc,
                         "prompt_qa": f"Scenario: {s} {sent}\nQuestion: {Q[task]}"})
with open("readout_R9k.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows)); print(rows[0]["prompt_qa"]); print(rows[-1]["prompt_qa"])
