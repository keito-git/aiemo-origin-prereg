# P2 Stage 1e plan (descriptive, registered before running): RepE anger directions vs d_word / d_sit

Registered before any Stage 1e output exists. Stage 1d (aa6523c) is still running for llm-jp and OLMo-2; Stage 1e runs after it.

## Question
Published emotion directions (RepE, Zou et al. 2023, arXiv:2310.01405) are built from contrasts that differ either in
the situation (RepE "concept" template: anger scenarios vs other emotions' scenarios, the word "anger" in every prompt)
or in an emotion word (RepE "function" template: "Act as if you are extremely {angry|furious|irritated}" vs
{pleased|calm|peaceful}). Where do these directions sit relative to d_word and d_sit, and do they move B2?

## Stimuli
RepE repository commit 5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e, data/emotions/*.json (MIT license), empty tags,
negatives sampled with seed 7300. Directions: difference of means, last token, layer L = round(0.33 x layers),
unit norm (the Stage 1d rule, not RepE's PCA). perp_word = the RepE direction with the span of the four d_word[X]
(Stage 1d, anger situation) removed.

## Outputs (all descriptive; no confirmatory claim)
- cos of each RepE direction with d_word[anger, fear, joy, eval] and d_sit; fraction of its norm in the word span.
- Delta_B2 of repe_concept, repe_function and their perp_word versions vs 1000 variance-matched random directions
  (seeds 7301-7304), same alpha and eigenbasis as Stage 1d. One-sided p reported per model, no pooling, no correction.

## Expectation stated in advance
If behavioural effects follow emotion words rather than situations (Stage 1c/1d), repe_function should be closer to
d_word than repe_concept is, and the perp_word versions should lose most of their B2 effect.
Models: Qwen3-8B (b968826d), llm-jp-4-8b-instruct (098f2b2c), OLMo-2-1124-7B-Instruct (470b1fba).
