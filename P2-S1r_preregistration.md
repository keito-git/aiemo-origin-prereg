# P2 Stage 1r: do emotion directions move judgments more than random directions of equal alignment?

Registered 2026-11-12, before the analysis below has been run on its confirmatory data (the Stage 1p data). The
Stage 1p runs exist (registered 491540f); this is a new analysis of existing data. What has been seen:
- Stage 1p: the projection z of each of the 16 directions on the shared judgment gradient correlated with its
  judgment effect zJ (r = .88 to .95; leave-one-readout-out 35/36 positive). The relation between projection and
  effect among the NULL (random) directions of Stage 1p, and the position of the 16 directions relative to that
  relation, have NOT been computed.
- Stage 1q (registered c15cd51; deviation note 95fec1f): H-align supported. Exploratory (s1q_explore2.py): among
  random directions, projection predicts effect with r = .84 to .92. Relative to that null line, the 12 Stage 1q
  directions showed a mean excess of .06 (SD units) over five models; Tulu showed +3.42 (12/12 positive). These 12
  directions are therefore NOT used for confirmation.
- A blind code-path run of s1r_excess.py on the Stage 1p data (output discarded unread, rc 0), and a run on the
  Stage 1q data that reproduced s1q_explore2.json.

## Question
1. Do emotion and magnitude directions move judgments by the amount that a random direction with the same alignment
   with the judgment gradient does, and no more?
2. Is Tulu an exception, as in Stage 1q?

## Data
Stage 1p: 16 directions (anger, joy, fear, sad, disg, grat, neu, numA, meaB, plnC, tmpD, dstE, wgtF, slfG, sizH,
qtyJ) x 100 variance-matched nulls each, six models, alpha = 0.2 x mean context norm. Gradient and effects use the
same 60 items for the directions and for the nulls.

## Analysis (s1r_excess.py)
- Per model and direction c: among the 100 nulls of c, y = mean over valid judgment readouts of the null effect,
  each readout z-scored over the nulls (L reversed); p = projection on the shared judgment gradient, z-scored over the
  nulls. Null line = OLS of y on p. excess_c = (zJ_c - line(projection z of c)) / SD of the null residuals.
- **H-excess (primary, equivalence).** Unit = model. The five models other than Llama-3.1-Tulu-3-8B. m = mean of the
  per-model mean excess. TOST with bounds +-0.5 (SD of null residuals), t with 4 df, alpha .05 each side.
  Equivalent if both one-sided p < .05 (90% CI inside +-0.5).
- **H-tulu (secondary, one-sided upper, alpha .05).** Mean excess of Tulu over the 16 directions > 0, one-sample t
  (15 df). The sign test is reported alongside.
- Descriptive: the TOST with Tulu included; per-model means and counts; category (7) vs magnitude (9) directions.

## Reading of the outcomes
- H-excess equivalent: in these five models, the judgment effect of emotion-related directions is what their
  alignment with the judgment gradient predicts for any direction. Emotional content adds no route of its own.
- H-excess not equivalent, mean > 0.5: emotion-related directions move judgments more than random directions of
  equal alignment. Mean < -0.5: less.
- H-excess inconclusive (CI crosses a bound): not decided; the bound and power are reported.
- H-tulu significant: Tulu is a confirmed exception (an emotion-specific excess in one model).

## Files
p2_stage1r/s1r_excess.py (imports s1o_rating.py in p2_stage1o/). Data: Stage 1p outputs (md5 list s1p_md5_all.txt).
