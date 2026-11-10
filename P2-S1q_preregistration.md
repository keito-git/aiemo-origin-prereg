# P2 Stage 1q: does alignment with the judgment gradient predict, in advance, the judgment effect of new directions?

Registered 2026-11-10, after the probes and the smoke on an unplanned model (Qwen3-8B-Base), and before any steered
Stage 1q run of a planned model. What has been seen before registration:
- All Stage 1p results (registered 491540f).
  - Registered tests: the judgment effect of 16 intensity directions was not explained by rated extremity
    (H-mag p = .55) or rated emotionality (H-emo p = .93). Emotion and magnitude directions did not differ (H-cat
    p = .92).
  - Rated unpleasantness (H-val, secondary, uncorrected): p = .040. It depended on two models; without Mistral and
    Tulu, p = .651.
  - Exploratory: the projection of each direction on the shared judgment gradient correlated with its effect within
    model (r = .88 to .95).
  - Exploratory, leave one readout out: the gradient of the other judgment readouts predicted the held-out readout
    in 35 of 36 model x readout cells (mean r = .58).
- Two probes of the Stage 1q closings (no steering, six models). In probe 1, shpT (sharp blade) was rated pleasant by
  two models and was replaced by grsT (greasy surface). In probe 2, the extremity difference between the unpleasant
  and pleasant groups was -.11 to .63 in every model (criterion: below 1). Every new contrast had the predicted sign
  of pleasantness at the group level.
- The smoke (Qwen3-8B-Base, 2 nulls per contrast, 18 min, rc 0, var_check 1.0). A code-path dry run of
  s1q_pooled_tests.py on two copies of the smoke output (--all_valid, 200 draws). Its values are not interpretable
  (two nulls, one model duplicated).
- No effect of any Stage 1q direction on any readout of a planned model has been looked at.

## Question
1. Does the alignment of a new direction with the judgment gradient predict its effect on the judgments? The
   alignment is computed from other items, before steering.
2. Do unpleasant magnitude directions make judgments harsher than pleasant ones of matched extremity?

## Directions (s1q_rating.py)
- Twelve new magnitude contrasts, none used before (stimuli/build_stage1q_closings.py, stage1q_closings.json).
  - Pleasant: brtK, swtL, sftM, frgN, qutP, clrQ.
  - Unpleasant: nsyR, stkS, grsT, btrU, cldV, drtW.
- Every contrast has the form "<object> was extremely <adj>." (hi) vs "<object> was slightly <adj>." (lo), with 12
  sentences each.
- Directions are unit(W_hi - W_lo), built exactly as in Stage 1p: same layer, same alpha, same covariance.

## Non-circularity
- **Items.** The 60 scenarios are split. The gradient uses only A (L01-L30); the effects and their nulls use only B
  (L31-L60).
- **Fixed in advance.** For every direction and every null, the projection on the A gradient is computed and saved
  before any steered forward pass of that model. Its SHA256 is logged (s1q_pred_*.json, s1q_proj_null_*.npy).

## Tests (s1q_pooled_tests.py)
- z of a readout = (B effect - null mean) / null SD. The null is 100 variance-matched random directions per contrast
  (seed 9100 + contrast index). The sign of L is reversed.
- zJ_B = mean z over the valid judgment readouts. Validity is taken from the Stage 1o pooled tests.
- Projection z = (projection on the shared A gradient - null mean) / null SD, using the same null directions.
- **H-align (primary, one-sided upper, alpha .05).**
  - Statistic: the mean over models of atanh(r), where r is the Pearson r of projection z and zJ_B over the 12
    contrasts of that model.
  - Null: projection z permuted over contrasts within each model (10000 draws, seed 9150).
- **H-val2 (secondary, one-sided lower, alpha .05).**
  - Statistic: the sum over models of [mean unpleasant - mean pleasant] of zJ_B, after regressing zJ_B on the
    standardized ext rating within model.
  - Null: group labels permuted within model.
- Descriptive: per-readout alignment r, including L; the OLS of zJ_B on ext, emo and val within model; zJ_B, zC_B
  and the ratings per contrast.
- A model enters if its Stage 1q run completed and it entered the Stage 1o tests. With fewer than two models, the
  result is "not testable".

## Reading of the outcomes
- H-align significant: the effect of a new direction can be predicted in advance from model-internal geometry, while
  the human classification axes (emotion category, rated extremity, rated emotionality) did not predict it in Stage
  1p.
- H-align not significant: the Stage 1p alignment result does not transfer to new directions and items. It is read
  as specific to the directions and items it was found on.
- H-val2 significant (negative): unpleasantness makes judgments harsher at matched extremity. This would be the one
  human-interpretable predictor that survives.
- H-val2 not significant: the Stage 1p H-val result is not confirmed.

## Files
p2_stage1q/: s1q_rating.py, s1q_pooled_tests.py, s1q_probe.py, queue_s1q.sh, stimuli/build_stage1q_closings.py,
stage1q_closings.json, SHA256SUMS. s1q_rating.py imports s1o_rating.py (in p2_stage1o/) and s1p_probe.py (in
p2_stage1p/). Models and revisions are as in Stage 1n.
