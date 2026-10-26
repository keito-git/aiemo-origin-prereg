# P2 Stage 1j preregistration: a rating readout with no option words

Registered before any steered Stage 1j readout is run (planned or smoke). Before registration, only baseline-only
(no steering) format probes were run to choose a valid readout format; they are reported here in full.
Inputs: Stage 1i results (973fca8) and the internal reviews of 10/27.

## Why
Stage 1i: B2m (options differing only in size) failed the judgment check in all six models (it tracks the
plausibility of the amount, not the case), so its primary tests were not testable. B2n (formal vs informal) moves
with pc1_neg (sum +2.71, p 0.001) but cannot separate a stricter judgment from a bias toward formal/heavy words (R3).
A readout with no option words at all removes wording valence and formality; an unrelated rating in the same format
measures any remaining bias toward large/heavy ratings.

## Pre-registration probes (baseline only, no steering; s1j_format_probe.py, s1j_format_probe2.py)
- Probe 1 (Qwen3-8B-Base + six planned models; readout_R9.csv): 1-9 ratings with swapped anchors. No model follows
  swapped anchors in any format (cont, qa, chat): e.g. reversal Spearman Mistral qa -0.85, OLMo qa -0.97, Qwen3 chat
  -0.81. The reversed scale is dropped.
- Probe 2 (readout_R9s.csv): each scenario gets a mild or a severe added sentence. Forward-scale qa format:
  severe - mild = +3.6 to +5.1 rating points in the six models, positive in 100% of pairs; object weight Spearman
  0.70-0.87. cont and chat formats are weaker (chat fails in llm-jp: +0.02, object rho -0.06). Option readouts on
  the same manipulation: B2 +2.6 to +3.8 (87-99% pairs positive), B2n +1.5 to +3.1, B2x -0.2 to +2.5 (43-65%),
  B2m +0.2 to +0.7 (56-63%). The qa format is chosen.

## Readout (s1j_rating.py, readout_R9s.csv from build_r9s.py)
"Scenario: <B2 scenario> <mild|severe sentence>\nQuestion: On a scale from 1 (no sanction at all) to 9 (the most
severe sanction), how severe a sanction does this case deserve? Answer with one digit.\nAnswer:" -> E[digit] from the
renormalised next-token probabilities of the nine digits. S = 200 items (100 B2 items x mild/severe). Control O = 100
items: "Object: <object>\nQuestion: On a scale from 1 (extremely light) to 9 (extremely heavy), how heavy is it? ..."
(50 everyday objects x 2 frames). Delta = mean over items of steered minus baseline E.

## Runs
Six Stage 1g models and revisions; directions rebuilt as in Stage 1g (cos check): pc1_neg, word_anger, word_eval,
word_poseval, plus val_eval = unit(word_eval - word_poseval) and pc1_val = pc1_neg minus its component along
unit(word_joy + word_poseval). Nulls: variance-matched random directions, each scored on S and O in the same pass
(index-paired), n 1000 for pc1_neg, word_eval, word_poseval, n 500 for word_anger, val_eval, pc1_val; seeds 8001-8007.
Queue (queue_s1j.sh): smoke on Qwen3-8B-Base (n_null 2) -> automatic check (code and readout validity; stop if
invalid) -> six models -> tests.

## Gate (per model)
Positive control p < 0.05 (odd S items: hidden state after " 8" minus after " 2"; measured on even S items; 200
nulls); severity effect > 1 rating point and positive in >= 80% of pairs; object Spearman with log weight > 0.5.

## Registered tests (s1j_pooled_tests.py)
Primary, one-sided upper, Holm over two:
- K1 = sum over gated models of Delta_S(pc1_neg): the deserved-sanction rating rises with no option words.
- K2 = sum [Delta_S(pc1_neg) - Delta_O(pc1_neg)] (paired nulls): it rises more than an unrelated heaviness rating.
Secondary (no correction): sum Delta_O(pc1_neg) (two-sided); sum Delta_S for word_eval, word_poseval (upper),
word_anger, val_eval, pc1_val (two-sided); sum [Delta_S(word_eval) - Delta_S(word_poseval)] (two-sided);
sum [Delta_S - Delta_O] for word_eval, word_poseval, val_eval, pc1_val (two-sided).
Descriptive: Delta_S within mild and within severe items.

## Reading
- K1 and K2 significant: negative-evaluation steering makes the sanction judgment stricter, beyond option wording and
  beyond a general bias toward large/heavy ratings.
- K1 significant, K2 not: a stricter rating, but a general large/heavy rating bias is not excluded.
- K1 not significant: no wording-free stricter judgment is detected; the Stage 1c-1i effects are then attributed to
  output-wording bias (valence congruence and formal/heavy words).
- word_poseval: no effect expected if the Stage 1h positive-direction effect was valence congruence (Stage 1i B2n).

## Note on earlier stages (from probe 2, descriptive)
B2x and B2m do not track the severity manipulation, unlike B2 and B2n. The Stage 1h H1 result (undecided) is
therefore read as uninformative about judgment.
