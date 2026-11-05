# P2 Stage 1p: does the judgment effect of an intensity direction follow its extremity or its emotion?

Registered 2026-11-05, after the smoke on an unplanned model (Qwen3-8B-Base) and before any steered Stage 1p run of a
planned model. What has been seen before registration:
- All Stage 1o results (registered 5546cdf). In particular, a non-emotional magnitude direction (meaB, a gauge reading
  near the top vs the middle of its range) moved the sanction readouts at least as much as the anger intensity
  direction (mean z over the valid judgment readouts: 1.42 vs 1.03). The anger direction spilled over less to the
  controls (.19 vs 1.10).
- Probe 1 (14 contrasts) and probe 2 (16 contrasts): no steering, ratings of the closing sentences alone on six
  models. These are the predictors used here.
- No effect of any new Stage 1p direction on any readout has been looked at.

## Question
Stage 1o found the effect of the anger direction across readout formats, but did not find it specific to anger.
Stage 1p asks whether the judgment effect of an intensity direction (hi minus lo closing) follows:
- how extreme the contrast is rated (ext);
- how emotional it is rated (emo);
- how unpleasant it is rated (val).
The answer is used over 16 directions per model.

## Directions (s1p_rating.py)
W_x = unit(mean last-token state of the Stage 1c texts ending in closing set x, minus the neutral_C texts). The layer,
alpha (0.2 x mean context norm) and covariance are as in Stages 1n and 1o. Direction = unit(W_hi - W_lo) for:
- emotion (6): anger, joy, fear, sad, disg, grat;
- magnitude (10): neu, numA, meaB, plnC, tmpD, dstE, wgtF, slfG, sizH, qtyJ.
Closings are in stimuli/stage1p_closings.json (built by stimuli/build_stage1p_closings.py) and in the earlier closing
files. sizH and qtyJ (pleasant magnitude) were added after probe 1, because ext and unpleasantness were collinear
there (r = -.59 to -.88). In probe 2 the VIFs of ext, emo and val are at most 1.6 in every model.

## Readouts, effects, nulls
- Readouts, items (L01-L60) and effect definitions are exactly as in Stage 1o. There are ten readouts; for L, the
  sign is reversed.
- Readout validity is taken from the Stage 1o pooled tests (same models and items, no steering).
- z of a readout = (effect - null mean) / null SD. The null is 100 variance-matched random directions per contrast
  (seed 8900 + contrast index).
- zJ = mean z over the valid judgment readouts; zC = mean z over the valid controls.
- Predictors: the hi - lo ratings of ext, emo and val, measured in the same run. They are standardized across the 16
  contrasts within each model.

## Tests (s1p_pooled_tests.py)
- **H-mag (primary, one-sided, alpha .05)**: the sum over models of the coefficient of ext in the within-model OLS
  zJ ~ 1 + ext + emo + val. Inference is Freedman-Lane (residuals of the reduced model permuted within each model;
  10000 draws; seed 8950).
- Secondary (no correction):
  - H-emo (two-sided): the same statistic for the emo coefficient.
  - H-val (two-sided): the same statistic for the val coefficient.
  - H-cat (two-sided): emotion minus magnitude contrasts in zJ, after regressing zJ on ext within model. Inference
    by permutation of the category labels.
  - H-sel (two-sided): as H-cat, with zJ - zC as the outcome.
- Descriptive: per-model coefficients; zJ, zC and the ratings per contrast; the L breakdown (z_L minus the mean z of
  the other valid judgment readouts) against the digit-control z.
- A model enters if its Stage 1p run completed and it entered the Stage 1o tests. With fewer than two models the
  result is "not testable".

## Reading of the outcomes
| H-mag | H-emo | Reading |
|---|---|---|
| significant | not significant | the judgment effect follows the extremity carried by a direction, not its emotion |
| significant | significant | both contribute |
| not significant | significant (positive) | an emotion-specific route; meaB in Stage 1o was an exception |
| not significant | not significant | not explained by the ratings; direction-specific |

H-val is read separately. A negative val coefficient means that unpleasant contrasts make judgments harsher, which
is an alternative to both readings above.

## Files
p2_stage1p/: s1p_rating.py, s1p_pooled_tests.py, s1p_probe.py, queue_s1p.sh, stimuli/build_stage1p_closings.py,
stage1p_closings.json, SHA256SUMS. The Stage 1o code imported by s1p_rating.py (s1o_rating.py) is in p2_stage1o/.
Models and revisions are as in Stage 1n.
