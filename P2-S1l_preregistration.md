# P2 Stage 1l preregistration: is the judgment shift specific to negative emotion, and does intensity carry it?

Registered before any steered Stage 1l run. Before registration:
- A probe of s1l_rating.py with --probe was run on the six planned models. It stops before any steering. It reports
  only the manipulation checks (each closing sentence rated on its own) and the cosines between directions. The
  results are in p2_stage1l/s1l_probe_*.json and are summarised below.
- A sdpa-vs-eager diagnostic in bf16 (Qwen3-8B, Mistral-7B; one fixed random direction, not a target direction) found
  the default attention deterministic and equal to eager within bf16 precision. Stages 1c-1k are not affected.
Inputs: Stage 1k results and internal reviews R2/R3 of 10/28.

## Why
Stage 1k: steering along pc1_neg (first axis of anger, fear and negative-evaluation word directions) raised the
sanction rating (K1k, Holm p 0.004). It raised it more than a location rating of the same text (K3, Holm p 0.004), so a
general digit bias does not explain it. But a positive-evaluation direction also gave a positive S - C contrast
(p 0.036, mostly from a drop in C). The pc1_neg - poseval difference was not significant (p 0.66), and its null was
wide because the two directions' nulls were drawn independently. R3: the strongest remaining alternative is a
non-emotional property of negative words (normativity, threat). Two questions follow:
1. Is the shift larger for negative than for positive emotion, when the null keeps the dependence between the two
   directions?
2. Does emotional intensity, with valence held fixed, shift the judgment?

## Directions (closing-sentence contrasts on the Stage 1c anger items, as in Stages 1d-1k)
- pc1_neg = first SVD axis of {word_anger, word_fear, word_eval} (Stage 1g-1k; cos with Stage 1g = 1.0 in the probe).
- pc1_pos = first SVD axis of {word_joy, word_grat, word_poseval}. word_grat is new: 12 gratitude closings
  (stage1l_closings.json), mirroring anger_L in form.
- int_neg = unit(word_anger - word_anger_lo). anger_lo is new: 12 low-intensity anger closings ("I was annoyed.",
  "I was mildly irritated.", ...), sentence by sentence mirroring anger_L ("I was furious.", "I was absolutely
  livid.", ...). Same word count as neutral_C[k].
- int_pos = unit(word_joy - word_joy_lo). joy_lo is new: 12 low-intensity positive closings ("I was pleased.",
  "I was content afterwards.", ...), mirroring joy_L.
- val_pc1 = unit(pc1_neg - pc1_pos) (secondary).
- Layer, alpha and the null covariance as in Stage 1k.

## Probe (no steering)
| model | strength anger_L - anger_lo (frac+) | strength joy_L - joy_lo (frac+) | pleasant pos - neg | cos pc1_neg, pc1_pos | cos int_neg, int_pos | cos pc1_neg, int_neg |
|---|---|---|---|---|---|---|
| Mistral-7B | +2.65 (1.0) | +1.86 (1.0) | +7.01 | 0.68 | 0.45 | 0.19 |
| granite-3.1-8b | +2.10 (1.0) | +2.00 (1.0) | +7.74 | 0.72 | 0.52 | 0.16 |
| Tulu-3-8B | +2.65 (1.0) | +1.85 (1.0) | +5.08 | 0.63 | 0.45 | 0.19 |
| Qwen3-8B | +3.37 (1.0) | +2.62 (1.0) | +6.81 | 0.75 | 0.50 | 0.18 |
| llm-jp-4-8b | +2.19 (1.0) | +1.76 (1.0) | +6.73 | 0.70 | 0.49 | 0.19 |
| OLMo-2-7B | +1.83 (1.0) | +1.93 (1.0) | +7.62 | 0.67 | 0.41 | 0.19 |

- Every model rates the high-intensity closings stronger than their low-intensity mirrors (12/12 pairs) and the
  positive triple far more pleasant than the negative triple.
- pc1_neg and pc1_pos share most of their direction (cos 0.63-0.75). A fair comparison needs a null that keeps this
  dependence (below).
- int_neg and int_pos share part of their direction (cos 0.41-0.52), an intensity component common to both valences.
  int_neg is nearly orthogonal to pc1_neg (0.16-0.19).

## Readout and metrics
As in Stage 1k: readout_R9k.csv (60 scenarios x mild/severe x {S sanction, C indoors/outdoors}), qa format. E =
expected digit, slope = OLS slope over k of [log p_steered(k) - log p_base(k)]. J(v) = dslope_S(v) - dslope_C(v) is
the judgment-specific shift on the saturation-free metric (K3 in Stage 1k).

## Nulls
- Paired null for pc1_neg vs pc1_pos (n = 1000, seed 8301). In each draw, r_neg is variance-matched to pc1_neg and r_pos
  to pc1_pos, and cos(r_neg, r_pos) is set equal to cos(pc1_neg, pc1_pos). The bisection is on the mixing weight of the
  two random draws, done in the eigenbasis. The smoke check requires the cosine error < 0.01 and the variance ratio
  within 1%.
- int_neg, int_pos, val_pc1: independent variance-matched draws, n = 500 each (seeds 8302-8304).
- Positive controls pos_82 and pos_sev as in Stage 1k, 200 each (seeds 8306-8307).
- The direction vectors are saved (s1l_dirs_*.npy), with cos(pos_sev, pc1_neg) (R2 of Stage 1k).

## Runs (queue_s1l.sh)
1. Smoke on Qwen3-8B-Base with tiny nulls.
2. Automatic check of the code (cos with 1g, variance match, pair cosine) and of the S and C readout validity. The
   queue stops if this check fails.
3. Run the six models (Stage 1g revisions).
4. Run the tests (s1l_pooled_tests.py).

## Gate (per model)
- As in Stage 1k: pos_82 or pos_sev p < 0.05; S severity effect > 1, positive in >= 80% of pairs; C rho with location
  > 0.5 and |C severity effect| < 1.
- Valence check: pleasantness of the positive triple minus the negative triple > 2.
- Intensity checks (per valence): rated strength high minus low > 0.5 and positive in >= 75% of the 12 pairs. L2 uses
  the models that pass the gate and the anger intensity check. J(int_pos) uses the models that pass the gate and the
  joy intensity check.
- A primary test needs at least two models. If only one primary test is testable, its p is doubled (Holm over two,
  with the other set to 1).

## Registered tests (s1l_pooled_tests.py)
Primary, one-sided upper, Holm over two:
- L1 = sum over gated models of [J(pc1_neg) - J(pc1_pos)], with the paired null. The judgment shift is larger for
  negative than for positive emotion.
- L2 = sum over gated models (anger intensity check) of J(int_neg). Negative emotional intensity, with valence held
  fixed, shifts the judgment.

Secondary (no correction):
- L1e = sum [dE_S(pc1_neg) - dE_S(pc1_pos)], upper; L1s = sum [dslope_S(pc1_neg) - dslope_S(pc1_pos)], upper.
- sum J(pc1_neg), upper (Stage 1k K3 replicated on the paired draws).
- sum J(pc1_pos) and sum dE_S(pc1_pos), two-sided.
- sum J(int_pos), two-sided (joy intensity check).
- sum dE_S(int_neg), upper; sum dE_S(int_pos), two-sided.
- sum J(val_pc1) and sum dE_S(val_pc1), two-sided.

## Reading
- L1 significant: the judgment shift depends on negative valence, beyond what the emotion directions share.
- L1 not significant and J(pc1_pos) positive: the shift comes from a component that negative and positive emotion
  words share (emotion presence). The claim must then be about emotion words in general, not negative affect.
- L1 not significant and J(pc1_pos) near zero: undecided (the difference may be too small to detect).
- L2 significant and J(int_pos) > 0: emotional intensity raises sanctions whatever the valence (an arousal account).
- L2 significant and J(int_pos) <= 0: intensity matters only for negative emotion.
- L2 not significant: intensity within anger does not carry the effect at this strength.
