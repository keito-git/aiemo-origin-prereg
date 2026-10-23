# P2 Stage 1d preregistration: does the behavioural effect of the emotion-word direction depend on the meaning of the word?

Date: 2026-10-23. Written and committed before any Stage-1d run on a planned condition.
Smoke tests were run only on unplanned conditions: (1) Qwen3-8B, fear situations, closing sets {fear, joy}, readout B4 (all directions) and readout B2 restricted to d_word[joy], 3 null draws, 2 bootstrap samples; (2) OLMo-2-1124-7B-Instruct, fear situations, closing set {fear}, readout B4, 2 null draws. Readout B4 is not used below, and d_word[joy] from fear situations is not used below. After the smoke tests, two code changes were made: a seed offset for readouts B1/B4 (bug found by the smoke test) and an adaptive upper bound in the variance-matching bisection (one smoke null draw was capped at 0.93 of the target variance).

## Background
Stage 1c (preregistration commit 724ec20): in Qwen3-8B and llm-jp-4-8b-instruct, the direction from adding one anger word to a closing sentence (d_word) shifted sanction judgement (B2) beyond variance-matched random directions (p = 0.029, 0.033), and the direction from the anger situation without emotion words (d_sit) did not (p = 0.48, 0.57). The difference d_word - d_sit against the paired variance-matched null was borderline (p = 0.075, 0.062). Fear x risk choice (B3) showed no effect of either direction. Internal reviewers asked whether the effect depends on the meaning of the word and whether the fear null result reflects the readout.

## Stimuli
- Situations: the Stage 1c items (stage1c_anger.jsonl, stage1c_fear.jsonl), unchanged.
- Closings (stage1d_closings.json): sets anger, fear, joy, eval and neutral C, 12 sentences each; sentence k of every set has the word count of C_k (k = i mod 12). anger and fear are the Stage 1c sets with k = 6 lengthened by one word to fix a word-count mismatch. joy contains one positive emotion word. eval is a negative moral evaluation of the event with no word from the forbidden emotion list E.

## Directions (layer L = round(0.33 x number of layers), last token, difference of means, unit norm)
- d_word[X] = mean(S+X, N+X) - mean(S+C, N+C), X in {anger, fear, joy, eval}
- d_sit = mean(S+O, S+C) - mean(N+O, N+C), O = closing set of the situation's own emotion

## Models
Qwen/Qwen3-8B (b968826d9c46dd6066d109eabc6255188de91218), llm-jp/llm-jp-4-8b-instruct (098f2b2cf33021eba19a6d3582aa3d071ccc0aff), allenai/OLMo-2-1124-7B-Instruct (470b1fba1ae01581f270116362ee4aa1b97f4c84).

## Runs (per model)
1. Anger situations, closing sets {anger, fear, joy, eval}; readout B2 for all five directions; readout B3 for d_word[anger] and d_sit; closing-cluster bootstrap (500) of Delta_B2(d_word[anger]) - Delta_B2(d_sit).
2. Fear situations, closing set {fear}; readouts B2 and B3 for d_word[fear] and d_sit; no bootstrap.
Intervention: alpha = 0.2 x mean residual norm at layer L (192 fixed contexts), added at every position. Delta_R = mean change in the two-option log-odds of readout R.

## Null
For every direction d and readout R: 1,000 random directions matched to the variance of d, with a seed fixed per direction and readout (base 7201-7206, +100 for B3). Draws are saved in order. The null for a difference between directions, or a sum over models, is the same difference or sum of the draws with the same index. One-sided p = (#null >= observed + 1) / 1001. Two-sided p = min(1, 2 min(p_upper, p_lower)).

## Gate (reported first)
Positive control pos_R (estimated on odd items of R, scored on even items, 200 null draws). A model whose positive control has p >= 0.05 on a readout is dropped from the sums for that readout.

## Primary test 1 (anger situations, B2)
T1 = sum over models of [Delta(d_word[anger]) - Delta(d_sit)]; supported if one-sided p < 0.05.

## Primary test 2 (anger situations, B2), Holm correction over the three contrasts
T2_X = sum over models of [Delta(d_word[anger]) - Delta(d_word[X])], X in {eval, fear, joy}; one-sided.
Interpretation: all three significant -> specific to the meaning of the anger word. fear and joy significant, eval not -> carried by negative evaluation, not by an emotion word as such. Only joy significant -> negative words in general. None significant -> any strong closing word (lexical salience). Other patterns are described as they are.

## Secondary (no correction, two-sided)
Fear situations x B2: sum over models of Delta(d_word[fear]) and of Delta(d_sit), each against its summed null. Anger situations x B3: the same for d_word[anger] and d_sit.

## Descriptive
Single-model p for every direction; split-half cosines; cosines between directions; closing-cluster bootstrap CI per model; sensitivity analyses on the saved activations.

## Code
s1d_word_specificity.py (runs), s1d_pooled_tests.py (all tests above). Helpers: s1c_word_vs_situation.py (Readout), s1_light_round.py.

## SHA256 (files in p2_stage1d/)
```
ec8554d9482664c7ea9f079b677cd4ed935d2d85c2e7966e705768181c470c94  contexts.csv
05711d5a5828b42641d379b9b67119312cd5aecd13364dae768ef3185f04a020  readout_B.csv
06d209ba8e01fd2c0ffab32cfbaa3afdc0b80a4d5e7773e2be605205202e3ac5  s1_light_round.py
2a094ad0d1731700d9848bbcbbe2ba494b69744162fd18f46210fe074daa55b7  s1c_word_vs_situation.py
11ea25ade83b230fa34b55a8acc921d841f040d809576046095f04136b241554  s1d_pooled_tests.py
eb98b741b6d270a77345f527c059dfbac2353cf1c72f6fb6ce9e563f4bbb4467  s1d_word_specificity.py
8b66589a5c8217940e9625011a57a17a80895af7f6b645fdfd9627417a51852b  stage1c_anger.jsonl
f209cb3e407f9444c5cc4b4e87124e438eb0910cf72acccdc8b0357f04edd771  stage1c_fear.jsonl
36e7086107abdfd88e56da402b0714a03b01835fcf65af38ae4991bce6c441d4  stage1d_closings.json
```
