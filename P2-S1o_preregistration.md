# P2 Stage 1o: is the effect of the anger intensity direction invariant across readouts of the same judgment?

Registered 2026-11-03, after the smoke on an unplanned model (Qwen3-8B-Base) and before any steered Stage 1o run of a
planned model. No effect of a Stage 1o direction (int_neg, int_neu, int_pos, pc1_neg) on a Stage 1o readout has been
looked at. What has been seen before registration (all without the test directions):
- Stage 1n results and the Stage 1n gradient decomposition (registered f401d8d, 25946cc, f8156ed), including the
  exploratory comparison of first-order predictions with measured effects.
- Probe 3 (no steering, K scenarios): validity of CAT_S2 and PAIR on six models.
- Probe 2 of the neutral pair (no steering): extremity and emotion ratings of three candidate neutral sets.
- Pilot 3 (Mistral, Qwen3, OLMo; K scenarios): gradient-based positive controls and 40 random directions on all ten
  readouts (the 10 x 10 control matrix).
- Probe 4 (no steering, the new L scenarios used here): validity of all ten readouts on six models.
- Smoke of s1o_rating.py on Qwen3-8B-Base (not a planned model; 3 null pairs, 2 minor nulls; 45 min) and a
  code-path run of s1o_pooled_tests.py on copies of the smoke output with synthetic nulls. Neither is evidence.
Inputs: external review 2 (11/02), internal review R3 of the design (11/03), external review 3 (11/03).

## Question and the claim it can support
Stage 1n showed that the anger intensity direction raises the 1-9 sanction rating S, but not invariantly across
readouts of the same judgment (HARSH weaker, leniency L split across models). Stage 1o asks whether the effect holds
across readouts that ask for the same sanction judgment in different formats (digit, 4-choice, pairwise) and in
different words, including two readouts that avoid every sanction word (CAT_S2, PAIR).
If it holds, the claim is limited to: an encoding-invariant effect consistent with a change in the underlying
judgment, that is, an effect not explained by a direct projection on one readout format, on the digit direction, or
on the sanction words. We do not claim to identify a latent judgment variable.

## Stimuli (stimuli/build_r9o_main.py -> readout_R9o_main.csv, 3360 rows)
60 new rule-violation scenarios L01..L60 (30 indoors, 30 outdoors) x the Stage 1j mild / severe sentence. Every row
is copied from the probe files, with only the scenario sentence replaced. Readouts:
- judgment: S, HARSH, L (1-9 digit; L reversed), CAT_S (4-choice, "Which response does this case call for?"),
  CAT_S2 (4-choice without sanction words, "What should happen next?"), PAIR (adjacent ranks of the CAT_S2 options,
  both orders); four letter rotations for every 4-choice readout;
- controls: U_bright, U_fam (1-9 digit), CAT_U (lighting), CAT_R (room), 4-choice.

## Directions, alpha, nulls (s1o_rating.py)
- Layer, alpha (0.2 x mean context residual norm) and covariance as in Stage 1n; directions built from the Stage 1c
  texts as in Stage 1n: int_neg (anger_L - anger_lo), int_pos (joy_L - joy_lo), pc1_neg (cos with Stage 1n checked).
- int_neu is new: meaB_hi - meaB_lo (a gauge reading near the top vs the middle of its range;
  stage1o_neu_closings.json). Neutral check fixed before registration as a property of the sentences: on the six-model
  mean of probe 2, the emotion difference of the meaB pair (.51) is less than half that of the anger pair (2.84). The
  per-model check is reported only (OLMo's emotion ratings do not separate emotion from extremity: meaB 2.26 vs anger
  1.91).
- Effect of a direction on a readout = item mean of the change of w . log p over the choice tokens (digit: OLS slope
  over k; 4-choice: slope over semantic rank; PAIR: log-odds of the heavier option).
- Nulls: 500 variance-matched random pairs (r_neg, r_neu) with cos matched to cos(int_neg, int_neu) (seed 8801;
  p floor 1/501 = .002); 50 draws each for int_pos and pc1_neg (descriptive). The pair count was reduced from 600 for
  computing time (about 3360 items per evaluation).

## Readout validity and model entry (s1o_pooled_tests.py)
Validity per model, no steering (scores per scenario x severity):
- S: severity effect > 1, frac_pos >= .8. HARSH: as S and rho with S > .7. L: severity effect < -.5,
  frac_neg >= .8, rho with S < -.5.
- CAT_S: effect > 1, rotation consistency > .8, rho with S > .7. CAT_S2: effect > .5, rotation consistency > .8,
  rho with S > .7 (values chosen after probe 3; on the independent probe 4 all six models pass; sensitivity with
  effect > .7 and > 1.0 reported). PAIR: effect > 1 log-odds, frac_pos >= .8, rho with S > .7 (order consistency
  reported only).
- U_bright, U_fam: |effect| < 1.2, rho(mild, severe) > .5, |rho with S| < .3 (Stage 1n). CAT_U, CAT_R: |effect| < .5,
  rotation consistency > .5, |rho with S| < .3.
- Gradient positive control, necessary condition for every readout: unit(mean gradient of the readout over the odd
  scenarios) applied at alpha, read out on the even scenarios, one-sided p < .05 against the pair_neg nulls on the
  even scenarios. In pilot 3 all 30 readout x model cells passed at alpha (z 6.6-33.5), so this check is not used to
  rank readouts; it only removes a readout that cannot respond to an intervention.
A model enters the tests if >= 3 judgment readouts are valid, with >= 1 digit and >= 1 choice readout, >= 1 control
is valid, and the anger checks of Stage 1n pass. Fewer than two entered models: "not testable".

## Tests
z = (effect - null mean) / null SD per model and readout (L sign-reversed); the null of a statistic is the same
statistic on null draw j, summed over models with the same j.
- **I1 (primary, one-sided, alpha .05)**: sum over entered models of min over valid judgment readouts of
  z(int_neg). This is a cross-model worst-readout statistic: it is not an intersection-union test of "positive in
  every model and readout"; a negative readout in one model can be outweighed by strong effects elsewhere.
- Secondary (no correction): I2 (two-sided) sum of zmin(int_neg) - zmin(int_neu), and the same without OLMo;
  I3 (one-sided) mean z over valid judgment readouts minus mean z over valid controls; I4 (one-sided) min z over the
  valid readouts without sanction words (CAT_S2, PAIR); I1 without L; I1 with PAIR from pairs 2-3 and 3-4 only
  (pair 1-2 does not separate in OLMo and llm-jp on K and L scenarios); K1' (one-sided) z of the shared
  judgment-readout residual of the gradient split (below).
- Descriptive: per model intersection-union p (max one-sided p over valid judgment readouts) and the number of
  models below .05; every (model, readout) with z < 0; z tables for all four directions; the 10 x 10 positive-control
  matrix; cos of each direction with the option-text direction (rank 4 minus rank 1 of CAT_S and CAT_S2) and its rank
  among the nulls; per-scenario effects.

Power (s1o_power_sim.py, readout correlations from the pilot-3 nulls): if every judgment readout moves by the same
true z, the power of I1 is .75 / .96 / 1.00 at z = 1 / 1.5 / 2 with 3 models and .94 / 1.00 / 1.00 with 6 models.

## Gradient split (secondary, descriptive except K1')
Mean gradients g_t of every readout over all items at delta = 0. Remove the span of the four control gradients;
sign-align the six judgment residuals r_t. Shared judgment-readout residual = mean of r_t; digit-specific = mean of
the digit residuals minus shared; choice-specific = mean of the choice residuals minus shared. Projections
alpha v . component for the four directions and all nulls. The shared residual is not taken to be the judgment
variable itself.

## Reading of the outcomes
| I1 | I4 | Reading |
|---|---|---|
| significant | significant | encoding-invariant effect consistent with a judgment change (no latent variable claimed) |
| not significant | significant | moves the readouts without sanction words but breaks in the digit readouts (digit component or L) |
| significant | not significant | unlikely; examined as a measurement problem (e.g. PAIR pair 1-2) |
| not significant | not significant | the Stage 1n effect is specific to the digit readouts (lexical or digit-format projection) |
I2 not significant: no specificity to anger over neutral intensity. Noted in advance from probe 4: CAT_U rotation
consistency is low in Mistral (.38) and granite (.26), CAT_R in OLMo (.16); Qwen3 U_fam follows severity (+1.08).

## Files
p2_stage1o/: s1o_rating.py, s1o_pooled_tests.py, s1o_power_sim.py, s1o_gradpos_pilot.py, s1o_main_probe.py,
stimuli/build_r9o_main.py, readout_R9o_main.csv, stage1o_neu_closings.json, queue_s1o.sh, SHA256SUMS.
Models and revisions as in Stage 1n.
