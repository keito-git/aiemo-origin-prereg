"""Stage 1p intensity contrasts (2026-11-05, before any Stage 1p registration). Stage 1o found that a non-emotional
magnitude direction (meaB) moved the sanction readouts at least as much as the anger intensity direction. Stage 1p
asks whether the judgment effect of an intensity direction follows its extremity or its emotion, over about ten
directions. New hi/lo closing pairs (12 each, same slots as the Stage 1d/1l emotion closings where possible):
  emotion   fear_lo (paired with the existing fear_L), sad_L/sad_lo, disg_L/disg_lo, grat_lo (with grat_L)
  magnitude tmpD (temperature), dstE (distance), wgtF (weight), objects described in the third person with
            "extremely"/"slightly"-type degree words (the same intensifier structure as anger_L/anger_lo)
  pleasant  sizH (large pleasant places), qtyJ (abundance of pleasant things): added after the first probe, where
            extremity and unpleasantness were collinear across the magnitude contrasts (r = -.59 to -.88)
  self      slfG, first-person non-emotional magnitude ("I walked a very long way"), to separate the first-person
            frame of the emotion closings from their emotion
Each magnitude hi/lo pair has the same word count (the emotion pairs follow the
slots of anger_L/anger_lo, which differ in length). Rated alone without steering by s1p_probe.py. Writes
stage1p_closings.json."""
import json
from pathlib import Path

fear_lo = ["I was uneasy.", "It made me slightly nervous.", "I was mildly worried.", "I noticed my unease rising.",
           "I was slightly wary of it.", "That left me uneasy.", "I was uneasy the whole evening.",
           "It made me wary all day.", "I was unsettled by it.", "I could not hide my unease.",
           "I was uneasy afterwards.", "It filled me with unease."]
sad_L = ["I was devastated.", "It made me so sad.", "I was absolutely heartbroken.", "I noticed my sorrow rising.",
         "I was really sad about it.", "That left me miserable.", "I was grieving the whole evening.",
         "It made me sad all day.", "I was crushed by it.", "I could not hide my sorrow.",
         "I was despondent afterwards.", "It filled me with despair."]
sad_lo = ["I was disappointed.", "It made me slightly sad.", "I was mildly downcast.", "I noticed my gloom rising.",
          "I was slightly sad about it.", "That left me downcast.", "I was glum for the whole evening.",
          "It made me glum all day.", "I was let down by it.", "I could not hide my disappointment.",
          "I was glum afterwards.", "It filled me with mild gloom."]
disg_L = ["I was disgusted.", "It made me so sick.", "I was absolutely revolted.", "I noticed my disgust rising.",
          "I was really repulsed by it.", "That left me nauseated.", "I was revolted the whole evening.",
          "It made me queasy all day.", "I was sickened by it.", "I could not hide my disgust.",
          "I was repulsed afterwards.", "It filled me with revulsion."]
disg_lo = ["I was put off.", "It made me slightly uncomfortable.", "I was mildly put off.",
           "I noticed my distaste rising.", "I was slightly put off by it.", "That left me uncomfortable.",
           "I was put off the whole evening.", "It made me uncomfortable all day.", "I was bothered by it.",
           "I could not hide my distaste.", "I was put off afterwards.", "It filled me with distaste."]
grat_lo = ["I was appreciative.", "It made me mildly appreciative.", "I was mildly appreciative.",
           "I noticed my appreciation rising.", "I was slightly appreciative of it.", "That left me appreciative.",
           "I was appreciative for the whole evening.", "It made me appreciative all day.", "I was glad of it.",
           "I could not hide my appreciation.", "I was appreciative afterwards.", "It filled me with appreciation."]
tmpD = [("The oven was extremely hot.", "The oven was slightly warm."),
        ("The water was boiling hot.", "The water was slightly warm."),
        ("The room was extremely hot that day.", "The room was slightly warm that day."),
        ("The pavement was scorching hot.", "The pavement was mildly warm."),
        ("The metal plate was extremely hot.", "The metal plate was slightly warm."),
        ("The soup was extremely hot to touch.", "The soup was slightly warm to touch."),
        ("The engine was running extremely hot.", "The engine was running slightly warm."),
        ("The sand was burning hot.", "The sand was slightly warm."),
        ("The kettle was extremely hot afterwards.", "The kettle was slightly warm afterwards."),
        ("The air was extremely hot outside.", "The air was slightly warm outside."),
        ("The stove top was extremely hot.", "The stove top was slightly warm."),
        ("The car seat was extremely hot.", "The car seat was slightly warm.")]
dstE = [("The station was extremely far away.", "The station was fairly close by."),
        ("The village was extremely far from town.", "The village was fairly near to town."),
        ("The shop was an extremely long walk.", "The shop was a fairly short walk."),
        ("The airport was extremely far away.", "The airport was fairly close by."),
        ("The office was extremely distant from home.", "The office was fairly near to home."),
        ("The beach was extremely far from here.", "The beach was fairly close to here."),
        ("The bridge was extremely far downstream.", "The bridge was fairly close downstream."),
        ("The school was an extremely long drive.", "The school was a fairly short drive."),
        ("The farm was extremely far up north.", "The farm was fairly close up north."),
        ("The hotel was extremely far from everything.", "The hotel was fairly close to everything."),
        ("The park was extremely far across town.", "The park was fairly close across town."),
        ("The lake was extremely far into the hills.", "The lake was fairly close into the hills.")]
wgtF = [("The crate was extremely heavy.", "The crate was slightly heavy."),
        ("The suitcase was extremely heavy to lift.", "The suitcase was fairly light to lift."),
        ("The stone was extremely heavy.", "The stone was fairly light."),
        ("The parcel was extremely heavy in total.", "The parcel was fairly light in total."),
        ("The table was extremely heavy to move.", "The table was fairly light to move."),
        ("The bucket was extremely heavy with sand.", "The bucket was fairly light with sand."),
        ("The door was extremely heavy to open.", "The door was fairly light to open."),
        ("The sack was extremely heavy with grain.", "The sack was fairly light with grain."),
        ("The machine was extremely heavy to carry.", "The machine was fairly light to carry."),
        ("The trunk was extremely heavy with books.", "The trunk was fairly light with books."),
        ("The log was extremely heavy.", "The log was fairly light."),
        ("The box of tools was extremely heavy.", "The box of tools was fairly light.")]
slfG = [("I walked an extremely long way.", "I walked a fairly short way."),
        ("I carried an extremely heavy load.", "I carried a fairly light load."),
        ("I waited an extremely long time.", "I waited a fairly short time."),
        ("I drove an extremely long distance.", "I drove a fairly short distance."),
        ("I climbed an extremely high hill.", "I climbed a fairly low hill."),
        ("I read an extremely long report.", "I read a fairly short report."),
        ("I wrote an extremely long list.", "I wrote a fairly short list."),
        ("I moved an extremely large table.", "I moved a fairly small table."),
        ("I filled an extremely big bag.", "I filled a fairly small bag."),
        ("I crossed an extremely wide road.", "I crossed a fairly narrow road."),
        ("I painted an extremely large wall.", "I painted a fairly small wall."),
        ("I swam an extremely long distance.", "I swam a fairly short distance.")]

sizH = [("The garden was extremely large.", "The garden was fairly small."),
        ("The park was extremely large and green.", "The park was fairly small and green."),
        ("The beach was extremely wide and sandy.", "The beach was fairly narrow and sandy."),
        ("The library was extremely big.", "The library was fairly small."),
        ("The orchard was extremely large.", "The orchard was fairly small."),
        ("The lawn was extremely wide.", "The lawn was fairly narrow."),
        ("The meadow was extremely broad.", "The meadow was fairly narrow."),
        ("The balcony was extremely spacious.", "The balcony was fairly compact."),
        ("The pool was extremely large and clean.", "The pool was fairly small and clean."),
        ("The window was extremely wide and bright.", "The window was fairly narrow and bright."),
        ("The kitchen was extremely roomy.", "The kitchen was fairly compact."),
        ("The playground was extremely large.", "The playground was fairly small.")]
qtyJ = [("The tree had an extremely large number of blossoms.", "The tree had a fairly small number of blossoms."),
        ("The table held an extremely large amount of food.", "The table held a fairly small amount of food."),
        ("The field had an extremely large number of flowers.", "The field had a fairly small number of flowers."),
        ("The shelf held an extremely large number of books.", "The shelf held a fairly small number of books."),
        ("The basket held an extremely large amount of fruit.", "The basket held a fairly small amount of fruit."),
        ("The sky had an extremely large number of stars.", "The sky had a fairly small number of stars."),
        ("The pond had an extremely large number of fish.", "The pond had a fairly small number of fish."),
        ("The jar held an extremely large amount of honey.", "The jar held a fairly small amount of honey."),
        ("The garden had an extremely large number of birds.", "The garden had a fairly small number of birds."),
        ("The shop had an extremely large number of toys.", "The shop had a fairly small number of toys."),
        ("The vine had an extremely large number of grapes.", "The vine had a fairly small number of grapes."),
        ("The box held an extremely large number of cookies.", "The box held a fairly small number of cookies.")]

out = {"note": __doc__, "fear_lo": fear_lo, "sad_L": sad_L, "sad_lo": sad_lo, "disg_L": disg_L, "disg_lo": disg_lo,
       "grat_lo": grat_lo}
for name, pairs in (("tmpD", tmpD), ("dstE", dstE), ("wgtF", wgtF), ("slfG", slfG), ("sizH", sizH), ("qtyJ", qtyJ)):
    out[f"{name}_hi"], out[f"{name}_lo"] = [h for h, _ in pairs], [l for _, l in pairs]
for k, v in out.items():
    if k != "note":
        assert len(v) == 12, k
for a, b in (("tmpD_hi", "tmpD_lo"), ("dstE_hi", "dstE_lo"),
             ("wgtF_hi", "wgtF_lo"), ("slfG_hi", "slfG_lo"),
             ("sizH_hi", "sizH_lo"), ("qtyJ_hi", "qtyJ_lo")):
    for h, l in zip(out[a], out[b]):
        assert len(h.split()) == len(l.split()), (h, l)
Path(__file__).with_name("stage1p_closings.json").write_text(json.dumps(out, indent=1))
print("wrote", len(out) - 1, "lists")
