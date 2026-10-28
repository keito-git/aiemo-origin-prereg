"""Stage 1l closing sentences (2026-10-28). Sentence k of every set has the word count of neutral_C[k] (Stage 1d).
anger_lo: low-intensity anger, mirroring anger_L (high intensity) sentence by sentence.
joy_lo:   low-intensity positive feeling, mirroring joy_L (high intensity).
grat_L:   gratitude, a positive other-directed emotion, mirroring anger_L in form; with joy_L and poseval_L it forms
          the positive triple used for pc1_pos, as anger_L / fear_L / eval_L form the negative triple for pc1_neg.
Writes stage1l_closings.json."""
import json

SETS = {
    "anger_lo": ["I was annoyed.", "It made me slightly irritated.", "I was mildly irritated.",
                 "I noticed my irritation rising.", "I was slightly annoyed about it.", "That left me irritated.",
                 "I was irritated for the whole evening.", "It made me irritated all day.", "I was annoyed by it.",
                 "I could not hide my irritation.", "I was irked afterwards.", "It filled me with annoyance."],
    "joy_lo": ["I was pleased.", "It made me mildly pleased.", "I was mildly pleased.",
               "I noticed my contentment rising.", "I was slightly pleased about it.", "That left me content.",
               "I was content for the whole evening.", "It made me content all day.", "I was pleased by it.",
               "I could not hide my contentment.", "I was content afterwards.", "It filled me with contentment."],
    "grat_L": ["I was grateful.", "It made me so grateful.", "I was deeply grateful.",
               "I noticed my gratitude rising.", "I was really grateful for it.", "That left me thankful.",
               "I was grateful for the whole evening.", "It made me grateful all day.", "I was touched by it.",
               "I could not hide my gratitude.", "I was thankful afterwards.", "It filled me with gratitude."],
}
ref = json.load(open("stage1d_closings.json"))["neutral_C"]
for k, v in SETS.items():
    assert len(v) == 12 and len(set(v)) == 12, k
    for i, s in enumerate(v):
        assert len(s.split()) == len(ref[i].split()), (k, i, s, ref[i])
out = {"note": __doc__.split("Writes")[0].strip()} | SETS
json.dump(out, open("stage1l_closings.json", "w"), indent=1)
print({k: len(v) for k, v in SETS.items()})
