# P2 Stage 1c preregistration: emotion word vs. situation as the source of an emotion direction's behavioural effect

Date: 2026-10-23. Written and committed before any Stage-1c run on a planned condition.
(A smoke test was run only on an unplanned condition: anger, readout B4, relative depth 0.67, 3 null directions, 2 bootstrap samples, to check that the code runs.)

## Background (exploratory, not confirmatory)
In Stage 1 (Qwen3-8B, anger, layer 18), the difference-of-means anger direction did not shift two behavioural readouts more than variance-matched random directions (B1 p = 0.30, B2 p = 0.33). In an exploratory Stage 1b grid (5 directions x 2 readouts x 4 layers x 2 scales = 80 cells), only the direction estimated from anger texts that contain an emotion word shifted readout B2 at layer 12 (p = 0.02 at both scales; same null set; not significant after correction). Stage 1c tests this with new stimuli and one preregistered test.

## Stimuli (files in this commit, SHA256 below)
- `stage1c_anger.jsonl`, `stage1c_fear.jsonl`: 170 items each. S = emotion-inducing first-person situation without emotion words; N = the same scene with the inducing element made neutral (word count within 3). Written by claude-opus-5-5; audited against the forbidden emotion-word list E (0 hits) and for length.
- `stage1c_closings.json`: 12 closings with one emotion word per emotion (L_k) and 12 neutral closings with the same word count (C_k). Item i uses k = i mod 12.
- Four versions per item: S+L, S+C, N+L, N+C. The last token is always the end of the closing.

## Directions (layer L, last token, difference of means, unit norm)
- d_word = mean(S+L, N+L) - mean(S+C, N+C)
- d_sit  = mean(S+L, S+C) - mean(N+L, N+C)

## Intervention and readout
- The scaled direction alpha * d is added to the output of decoder layer L at every position. alpha = 0.2 x mean residual norm at layer L over 192 fixed contexts.
- Delta_R(d) = mean over readout items of [log p(target) - log p(other)] with the intervention minus the same without it. Readouts are the existing two-option items (`readout_B.csv`, unchanged from Stage 1).

## Primary test (one test)
- Model Qwen/Qwen3-8B (revision b968826d9c46dd6066d109eabc6255188de91218), emotion anger, relative depth 0.33 (layer 12), readout B2 (all 100 items).
- H1c: Delta_B2(d_word) - Delta_B2(d_sit) > 0.
- Test: item bootstrap, 1,000 resamples of the 170 items (seed 7104); both directions are re-estimated in each resample. H1c is supported if the one-sided 95% lower bound of the bootstrap distribution is > 0.

## Secondary (reported, not used for the primary decision)
- Each direction vs. 1,000 random directions matched to the variance of that direction (seeds 7101, 7102); one-sided empirical p = (#null >= observed + 1) / 1001.
- Interpretation table:
  - H1c supported, p_word < 0.05, p_sit >= 0.05 -> the behavioural effect is carried by the emotion word.
  - H1c not supported, both p < 0.05 -> the situation also carries the effect.
  - both p >= 0.05 -> the Stage 1b layer-12 finding does not replicate.
- Quality gates, reported first: split-half cosine of each direction (seed 7105); positive control pos_B2 (estimated on odd B2 items, scored on even items vs. 200 random directions matched to its variance). If pos_B2 has p >= 0.05, the primary result is not interpreted.

## Generality (same procedure, run after the primary test)
1. Fear, readout B3, relative depth 0.33. Predicted sign: negative (fear lowers the choice of the risky option, which is the target option of B3). Test: Delta_B3(d_sit) - Delta_B3(d_word) > 0 under the same bootstrap rule, i.e., d_word moves B3 more in the negative direction. Run only if pos_B3 passes the positive-control gate.
2. llm-jp/llm-jp-4-8b-instruct (revision 098f2b2cf33021eba19a6d3582aa3d071ccc0aff), anger, B2, relative depth 0.33, same test as the primary.
- No correction is applied across the generality conditions; each is reported with its own result.

## Descriptive only
Layer scan over relative depths 0.17 to 0.94; d_int; cosines between directions.

## Code
`s1c_word_vs_situation.py` (this commit). Shared helpers in `s1_light_round.py` (Stage 1).

## SHA256 (files in p2_stage1c/)
```
ec8554d9482664c7ea9f079b677cd4ed935d2d85c2e7966e705768181c470c94  contexts.csv
05711d5a5828b42641d379b9b67119312cd5aecd13364dae768ef3185f04a020  readout_B.csv
06d209ba8e01fd2c0ffab32cfbaa3afdc0b80a4d5e7773e2be605205202e3ac5  s1_light_round.py
2a094ad0d1731700d9848bbcbbe2ba494b69744162fd18f46210fe074daa55b7  s1c_word_vs_situation.py
8b66589a5c8217940e9625011a57a17a80895af7f6b645fdfd9627417a51852b  stage1c_anger.jsonl
440de341184d7c0540bd5c2cdbcb1cdf9393c4807160996c58f84a9445c1b001  stage1c_closings.json
f209cb3e407f9444c5cc4b4e87124e438eb0910cf72acccdc8b0357f04edd771  stage1c_fear.jsonl
```
