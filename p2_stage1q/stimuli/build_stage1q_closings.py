"""Stage 1q intensity contrasts (2026-11-10, before any Stage 1q registration). Twelve new non-emotional magnitude
contrasts, none used in Stages 1n-1p, for a prospective test of the gradient-alignment finding of Stage 1p and a
confirmation of H-val. All follow the tmpD form: a third-person description of an object, "extremely" (hi) vs
"slightly" (lo), the same word count within each pair.
  pleasant   brtK (bright sunny day), swtL (sweet fruit), sftM (soft blanket), frgN (fragrant flowers),
             qutP (quiet library), clrQ (clear water)
  unpleasant nsyR (loud noise), stkS (bad smell), grsT (greasy surface), btrU (bitter taste), cldV (freezing cold),
             drtW (dirt). grsT replaced shpT (sharp blade) after the probe, where shpT was rated pleasant by two
             models (val +1.0, +2.4); v1 is kept as stage1q_closings_v1_shpT.json
Rated alone without steering by s1q_probe.py. Writes stage1q_closings.json."""
import json
from pathlib import Path

NOUNS = {
    "brtK": ("bright", ["The morning", "The sky", "The beach", "The garden", "The meadow", "The afternoon",
                        "The terrace", "The hillside", "The courtyard", "The lake shore", "The park", "The day"]),
    "swtL": ("sweet", ["The peach", "The mango", "The strawberry", "The melon", "The grape", "The cherry",
                       "The pear", "The plum", "The orange", "The apricot", "The fig", "The pineapple"]),
    "sftM": ("soft", ["The blanket", "The pillow", "The towel", "The sweater", "The scarf", "The rug", "The cushion",
                      "The robe", "The quilt", "The mattress", "The kitten's fur", "The slipper"]),
    "frgN": ("fragrant", ["The rose", "The jasmine", "The lilac", "The garden", "The lavender", "The bouquet",
                          "The orchard", "The magnolia", "The gardenia", "The honeysuckle", "The peony", "The meadow"]),
    "qutP": ("quiet", ["The library", "The reading room", "The garden", "The chapel", "The study", "The forest",
                       "The bedroom", "The lake", "The museum", "The street", "The cabin", "The morning"]),
    "clrQ": ("clear", ["The water", "The lake", "The stream", "The pool", "The spring", "The bay", "The river",
                       "The pond", "The sea", "The fountain", "The lagoon", "The glass of water"]),
    "nsyR": ("loud", ["The drill", "The engine", "The alarm", "The siren", "The machine", "The traffic", "The horn",
                      "The generator", "The jackhammer", "The fan", "The compressor", "The motor"]),
    "stkS": ("foul-smelling", ["The drain", "The bin", "The sewer", "The fridge", "The alley", "The basement",
                               "The dumpster", "The sink", "The cellar", "The gutter", "The compost", "The tank"]),
    "grsT": ("greasy", ["The pan", "The counter", "The stove", "The plate", "The handle", "The tray",
                        "The grill", "The table", "The floor", "The oven door", "The rag", "The window"]),
    "btrU": ("bitter", ["The coffee", "The medicine", "The tea", "The root", "The syrup", "The tonic", "The pill",
                        "The herb", "The drink", "The rind", "The extract", "The broth"]),
    "cldV": ("freezing", ["The wind", "The water", "The room", "The night", "The air", "The floor", "The lake",
                          "The garage", "The morning", "The cellar", "The rain", "The metal rail"]),
    "drtW": ("dirty", ["The floor", "The carpet", "The sink", "The window", "The kitchen", "The bathroom",
                       "The car", "The table", "The jacket", "The wall", "The bench", "The stairs"]),
}
PLEASANT = ("brtK", "swtL", "sftM", "frgN", "qutP", "clrQ")
UNPLEASANT = ("nsyR", "stkS", "grsT", "btrU", "cldV", "drtW")


def build():
    out = {"note": __doc__}
    for k, (adj, nouns) in NOUNS.items():
        assert len(nouns) == 12 and len(set(nouns)) == 12, k
        out[f"{k}_hi"] = [f"{n} was extremely {adj}." for n in nouns]
        out[f"{k}_lo"] = [f"{n} was slightly {adj}." for n in nouns]
        assert all(len(h.split()) == len(l.split()) for h, l in zip(out[f"{k}_hi"], out[f"{k}_lo"])), k
    return out


if __name__ == "__main__":
    o = build()
    p = Path(__file__).with_name("stage1q_closings.json")
    assert not p.exists(), f"{p} exists; refuse to overwrite"
    p.write_text(json.dumps(o, indent=1, ensure_ascii=False))
    print(p, len(o) - 1, "lists")
