# P2 Stage 1m preregistration: does the intensity effect make the judgment harsher, and is it emotional?

Registered before any steered Stage 1m run. Before registration:
- A probe of s1m_rating.py with --probe was run on the six planned models. It stops before any steering. It reports
  only the manipulation checks (each closing sentence rated on its own), the cosines between directions and the
  baseline readout checks. The results are in p2_stage1m/s1m_probe_*.json and are summarised below.
- Two gate rules were set after the probe (below). The probe reports no steered outcome.
Inputs: Stage 1l results and internal reviews R2/R3 of 10/29.

## Why
Stage 1l: the judgment shift from emotion directions was not specific to negative emotion (L1, Holm p 0.23). An
intensity direction with valence held fixed (int_neg = furious - irritated) raised the sanction rating more than a
location rating of the same text (L2, Holm p <= 0.004). The joy intensity direction did the same (p_two 0.020). Two
alternatives remain (R2, R3):
A. Linguistic intensity in general, not emotional intensity. "furious" vs "irritated" also differ in how extreme the
   wording is.
B. A bias towards "more of whatever is asked", not a harsher judgment. The sanction question asks "how severe"; an
   intensity direction may raise any rating of "how much".

## Readout (readout_R9m.csv)
- The Stage 1k items: 60 scenarios x mild/severe, task S (deserved sanction, 1-9) and task C (indoors/outdoors, 1-9),
  unchanged.
- New task L: the same scenario text with a leniency question, "On a scale from 1 (not lenient at all) to 9 (extremely
  lenient), how leniently should this case be treated?". A higher digit means a milder response.
  - Harsher judgment (prediction): S up, L down.
  - Alternative B: S up, L up.
- E = expected digit; slope = OLS slope over k of [log p_steered(k) - log p_base(k)], as in Stages 1k-1l.
- J_S(v) = dslope_S(v) - dslope_C(v) (the Stage 1l J). J_L(v) = dslope_L(v) - dslope_C(v).

## Directions (closing-sentence contrasts on the Stage 1c anger items, as in Stages 1d-1l)
- int_neg = unit(word_anger - word_anger_lo); int_pos = unit(word_joy - word_joy_lo). Same as Stage 1l (cos 1.0 in
  the probe).
- int_neu = unit(word_neu_hi - word_neu_lo). New: a non-emotional intensity contrast (stage1m_closings.json). There
  are 12 pairs about size, height, weight, amount and weather ("It was large." / "It was enormous.", "The bag was
  slightly heavy too." / "The bag was extremely heavy too."). They mirror the anger pairs in form where possible and
  have the word count of neutral_C[k].
- int_com = unit(int_neg + int_pos) and int_dif = unit(int_neg - int_pos): the shared and valence-specific parts of
  intensity (R3).
- pc1_neg, pc1_pos as in Stage 1l (cos 1.0 in the probe).
- Layer, alpha and the null covariance as in Stages 1k-1l.

## Probe (no steering)
| model | L severity effect (frac neg) | rho(S, L) | extremity: anger / joy / neutral hi - lo | emotion: anger hi - lo | emotion: neutral hi - lo | cos(int_neg, int_neu) |
|---|---|---|---|---|---|---|
| Mistral-7B | -3.43 (1.00) | -0.88 | +2.66 / +2.34 / +3.27 | +2.74 | +1.02 | 0.43 |
| granite-3.1-8b | -0.98 (0.98) | -0.74 | +3.39 / +4.35 / +5.08 | +3.61 | +0.98 | 0.48 |
| Tulu-3-8B | -2.40 (0.98) | -0.85 | +2.82 / +2.34 / +3.33 | +2.81 | +1.51 | 0.36 |
| Qwen3-8B | -2.67 (0.93) | -0.77 | +2.72 / +2.54 / +3.68 | +3.37 | +1.00 | 0.43 |
| llm-jp-4-8b | -2.18 (1.00) | -0.90 | +2.64 / +2.32 / +2.74 | +2.56 | +1.14 | 0.44 |
| OLMo-2-7B | -1.26 (0.92) | -0.71 | +2.09 / +2.91 / +3.36 | +1.90 | +2.11 | 0.34 |

- The leniency rating follows the scenario severity in the opposite direction to the sanction rating in every model
  (rho -0.71 to -0.90). granite rates leniency near the floor (2.44 mild, 1.46 severe), so its effect is smaller.
- The neutral pair is rated at least as extreme as the anger pair in every model.
- The neutral high sentences are also rated somewhat more emotional than the low ones (+0.98 to +2.11). Extreme
  wording carries some emotion for these models. In four models (Mistral, granite, Qwen3, llm-jp) the neutral emotion
  difference is less than half of the anger one; in Tulu (0.54) and OLMo (1.11) it is not.

## Rules set after the probe (no steered data seen)
1. L validity in the gate: L severity effect < -0.5 (not -1), L lower for the severe version in >= 80% of pairs, and rho(S, L) < -0.5.
   Reason: granite's leniency ratings sit near the floor.
2. Neutral emotion check: the neutral emotion difference must be less than half of the anger emotion difference (not
   an absolute bound of 1). Reason: no absolute bound separates the models usefully; the relative bound states the
   point of the control (much less emotional than the anger pair).

## Nulls
- Paired null for int_neg vs int_neu (n = 1000, seed 8401). r_neg is variance-matched to int_neg and r_neu to int_neu,
  and cos(r_neg, r_neu) is set equal to cos(int_neg, int_neu). A draw whose cosine misses the target by more than 0.01
  is redrawn (fixes the Stage 1l deviation); the number of redraws is recorded. The queue checks the maximum error in
  every model.
- int_pos: 500 independent variance-matched draws (seed 8402). int_com, int_dif, pc1_neg, pc1_pos: 300 each (seeds
  8403-8406).
- Positive controls pos_82 and pos_sev on S, 200 each (seeds 8407-8408).

## Gate (per model)
- As in Stage 1l: pos_82 or pos_sev p < 0.05; S severity effect > 1, positive in >= 80% of pairs; C rho with location
  > 0.5 and |C severity effect| < 1; valence check > 2.
- L validity (rule 1).
- Anger checks: feeling strength (Q_INT) and extremity (Q_EXT) high minus low > 0.5, positive in >= 75% of the 12 pairs.
- Neutral checks: extremity high minus low > 0.5, positive in >= 75%, and rule 2.
- M1 uses the models that pass the gate and the anger checks. M2 uses the models that also pass the neutral checks.
  A primary test needs at least two models; if only one is testable, its p is doubled.

## Registered tests (s1m_pooled_tests.py)
Primary, one-sided, Holm over two:
- M1 = sum of J_L(int_neg), lower tail. The anger intensity direction lowers the leniency rating relative to the
  location rating: the judgment becomes harsher. The null is the r_neg half of the paired null.
- M2 = sum of [J_S(int_neg) - J_S(int_neu)], upper tail, paired null. Emotional intensity raises the sanction more than
  non-emotional intensity.

Secondary (no correction):
- sum J_S(int_neg), upper (Stage 1l L2 again; same direction and items, so not an independent replication).
- sum dslope_L(int_neg), lower.
- sum J_L(int_pos), two-sided (joy checks).
- sum J_S(int_neu) and sum J_L(int_neu), two-sided (neutral checks).
- sum [J_L(int_neg) - J_L(int_neu)], two-sided, paired null.
- M2 over all models that pass the neutral extremity check, regardless of rule 2 (upper).
- sum J_S and sum J_L of int_com, int_dif, pc1_neg, pc1_pos, two-sided.

## Reading
- M1 significant: the intensity effect is a harsher judgment, not "more of whatever is asked". If instead sum J_L(int_neg)
  is positive and its two-sided p < 0.05, alternative B holds and the Stage 1l result is a response bias.
- M2 significant: the effect is larger for emotional intensity than for non-emotional intensity of matched extremity.
- M2 not significant and J_S(int_neu) positive: linguistic intensity in general carries the effect (alternative A).
  The claim must then be about intensity of wording, not emotion.
- M2 not significant and J_S(int_neu) near zero: undecided.
- J_S(int_com) vs J_S(int_dif): whether the shared intensity part or the valence-specific part carries the effect.

## Runs (queue_s1m.sh)
1. Smoke on Qwen3-8B-Base with tiny nulls.
2. Automatic check of the code (cos with Stage 1l, variance match, pair cosine error <= 0.01) and of the S, C and L
   readout validity (for the unplanned base model, L only needs the right sign). The queue stops if this check fails.
3. The six models (Stage 1g revisions). After each model the queue prints the pair-cosine check.
4. The tests (s1m_pooled_tests.py).
