# P2 Stage 1s: which post-training stage produces the residual beyond alignment in Tulu?

Registered 2026-11-13, before any main-run steering of the five checkpoints below. Design: P2 02_設計/15_段階1s_..._draft_v2_20261112.md
(internal reviews R2/R3 applied).

## Background (registered results this builds on)
- Stage 1r (d0c0d9a), H-tulu, confirmed on the Stage 1p data: in Llama-3.1-Tulu-3-8B (final, after RLVR), the mean
  excess over the 16 directions was +1.56, t(15) = 4.76, p = .0001, 14/16 positive. In the five other models, a large
  positive excess was not supported (H-excess TOST inconclusive).
- Question: is this residual already present after SFT, after DPO, or only in the final model? Does it depend on the
  Tulu 3 recipe (OLMo 2 got the same recipe on another base) or on the Llama-3.1-8B base (Meta's own post-training)?

## What has been seen before registration
- Smoke run (10 nulls per contrast, 20 null pairs for validity) on Tulu-3-8B-SFT and Tulu-3-8B-DPO:
  - Readout validity (Stage 1o criteria, s1s_validity.py): both checkpoints enter. SFT: valid judgment readouts S,
    HARSH, CAT_S, CAT_S2, PAIR (L failed rho with S = -0.50 vs the criterion < -0.5); DPO: all six. Valid controls in
    both: U_fam, CAT_R (the same as the final Tulu in Stage 1o). Positive controls beat all 20 null pairs.
  - Cosine of the 16 directions between SFT, DPO and the final model: median .998 to 1.000.
  - s1s_excess.py ran to completion with finite values. **The smoke excess values were written to a file and not
    opened** (excess_smoke_SEALED.json). They are not used.
- Code checks: s1s_validity.py reproduces the Stage 1o entry decisions and valid readouts of all six Stage 1o models
  exactly (s1o_pooled_tests.json). s1s_excess.py reproduces the registered Stage 1r excess values of all six models x
  16 directions (max abs difference 1e-14). s1s_tests.py was checked on relabeled Stage 1p values.
- No main-run output exists for any of the five checkpoints.

## Checkpoints (Hugging Face, revisions fixed)
| role | model | revision |
|---|---|---|
| stage axis | allenai/Llama-3.1-Tulu-3-8B-SFT | f2a0b46b0cfda21003c6141b1ff837b7e165524d |
| stage axis | allenai/Llama-3.1-Tulu-3-8B-DPO | a7beb67e33ffd01cc87ac3b46cadc1000985b8db |
| OLMo axis (same recipe, other base) | allenai/OLMo-2-1124-7B-SFT | 1de02c0175118a9de5854aec80a1f970e701e928 |
| OLMo axis | allenai/OLMo-2-1124-7B-DPO | e34ea60adff2e575f4fe7569eaffd1b28509b6fd |
| Llama axis (same base, other recipe) | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |

The final models (Tulu-3-8B, OLMo-2-1124-7B-Instruct) were run in Stage 1p; their excess (Stage 1r) is listed as
description only. All five checkpoints are computed in one queue; stage 2 (OLMo, Llama axes) is registered here,
before any stage 1 result is seen.

## Method (unchanged from Stages 1o, 1p, 1r; only the model changes)
- s1o_rating.py per checkpoint: base scores of the ten readouts and the gradient positive controls; 500 null pairs,
  50 minor nulls (as Stage 1o).
- s1p_rating.py per checkpoint: the 16 directions (anger, joy, fear, sad, disg, grat, neu, numA, meaB, plnC, tmpD,
  dstE, wgtF, slfG, sizH, qtyJ) rebuilt in each checkpoint from the same sentence pairs; 100 variance-matched nulls
  per direction (seeds as Stage 1p); layer round(0.33 L) - 1; alpha = 0.2 x mean context norm.
- Readout validity (s1s_validity.py): the Stage 1o rule unchanged (validity(), gradient positive control p < .05
  against the pair_neg nulls on the even scenarios, entry rule: >= 3 valid judgment readouts with >= 1 digit and
  >= 1 choice readout, >= 1 valid control, anger check). A checkpoint that does not enter is reported as **not
  measurable**, never as "no effect", and is dropped from its Holm family.
- Excess (s1s_excess.py): exactly as s1r_excess.py (null OLS line of zJ on the projection z on the shared judgment
  gradient; excess = distance from the line in SD of the null residuals), with the checkpoint's own valid readouts.

## Tests (s1s_tests.py)
- **H-stage (primary).** For c in {Tulu-SFT, Tulu-DPO}: mean excess over the 16 directions > 0, one-sided t (df 15).
  Holm over the two. "Positive" = Holm-adjusted p < .05; otherwise "not positive".
- **Zero (secondary, per checkpoint).** TOST of the mean excess with bounds +-0.5 (df 15). "No residual at this
  stage" is written only if TOST p < .05. Not positive and TOST not significant = "undecided".
- **H-recipe (secondary, two separate families).** OLMo axis: OLMo-SFT and OLMo-DPO, one-sided t, Holm over the two.
  Llama axis: Llama-3.1-8B-Instruct, one-sided t.
- Power (computed by code, noncentral t, n = 16): 80% needs d = .75 at the Holm alpha .025 (d = .65 at .05). With the
  between-direction SD of the final Tulu (1.31), that is a mean excess of about 0.98, about two thirds of the final
  Tulu value. Smaller residuals (0.5 to 1.0) can be missed. TOST power is low: with a true mean of 0, it declares
  equivalence in about 4% (SD 1.31) to 33% (SD 0.93) of runs. **"Undecided" is therefore the expected reading for a
  stage without a large residual**; "no residual" claims are unlikely.

## Reading of H-stage (fixed now)
| SFT | DPO | reading |
|---|---|---|
| positive | positive | The residual arises at SFT and survives DPO; it is attributed to SFT data or procedure. |
| positive | not positive | Residual at SFT, not positive after DPO, present again in the final model (non-monotonic). No one-directional stage claim; differences are described as exploratory. |
| not positive | positive | The residual arises at DPO (preference learning). |
| not positive | not positive | Suggests the residual arises at the final RLVR stage. "Arises at RLVR" is written only if TOST shows equivalence for both SFT and DPO. |

If one of the two Tulu checkpoints is not measurable, the stage axis is reported as not fully measurable and only the
measurable checkpoint is read. If both are not measurable, the stage axis is not testable.

## Reading of H-recipe
- Positive in Tulu and in the OLMo axis: consistent with the Tulu 3 recipe. Positive in Tulu and in Llama-Instruct:
  consistent with the Llama-3.1-8B base. Positive in Tulu only: a base x recipe combination. These readings are
  secondary and are written as "consistent with".

## Exploratory (not confirmatory)
- Differences between adjacent stages (DPO - SFT, final - DPO): the scale of the excess can differ across
  checkpoints, so differences are described only.
- Category (7) vs magnitude (9) directions within each checkpoint with a residual.
- Diagnostics: slope and residual SD of the null line per checkpoint and direction; cosine of the directions across
  stages; the excess recomputed with fewer valid judgment readouts (stability by readout count).
- H-data (Tulu SFT data-source ablations): run only if Tulu-SFT is positive under H-stage; exploratory, since content
  and amount of data change together. Not run otherwise.

## Strongest alternative
"The Tulu residual is model x post-training noise, not a systematic effect." Addressed by the registered tests here,
the null-line diagnostics, and whether the SFT -> DPO -> final sequence changes consistently.

## Files
p2_stage1s/: s1s_validity.py, s1s_excess.py, s1s_tests.py, queue_s1s.sh, sha256 list. Imports s1o_rating.py
(p2_stage1o/), s1p_rating.py (p2_stage1p/), s1o_pooled_tests.py (p2_stage1o/).
