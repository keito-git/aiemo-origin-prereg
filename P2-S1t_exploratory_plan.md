# P2 Stage 1t: registered outcome ("not measurable") and an exploratory S+HARSH-only comparison

Written 2026-11-16, before any base-model output of the main s1p run (`/root/aiemo/results/p2_stage1t_p`) was opened,
and before any S+HARSH-only excess was computed for any checkpoint.
Design: P2 02_設計/16_段階1t_..._draft_20261116.md. Report: P2 04_報告/段階1t_試し実行_結果報告_20261116.md
(internal reviews R2/R3 applied).

## 1. Registered outcome of Stage 1t (final; not changed by anything below)
- Design §3.3 fixed the reading in advance: a base model whose readouts fail the Stage 1o entry rule is "not measurable".
- Smoke run (11/16, 20 null pairs, 5 minor nulls, 10 s1p nulls; the item set is the full set, so no-steer validity
  would not change in a main run):
  - Llama-3.1-8B (d04e592b) and OLMo-2-1124-7B (7df9a825): valid judgment readouts S and HARSH only. No choice-format
    readout (CAT_S, CAT_S2, PAIR) passed. Both fail the entry rule.
  - The gradient positive control gave p = .048 on all readouts. This is the minimum attainable p with 20 null pairs,
    so it does not discriminate. The entry decision was made entirely by the item-based no-steer criteria.
  - Choice-format severity effects (criterion > 1; CAT_S2 > 0.5): Llama base CAT_S 0.70, CAT_S2 0.47, PAIR 0.64; OLMo base
    0.31, 0.22, 0.33. PAIR and CAT_S are far below 1, so the decision does not hinge on CAT_S2 being near its threshold.
    OLMo L (−0.49 vs < −0.5) would not change the decision, because L is a rating-format readout.
  - s1s_excess.py wrote only {"status": "not measurable"} for both. No excess value exists for the smoke run.
- Therefore: H-base "not measurable" for both; registered main run not done; H-shift not testable; H-data not started.
  The paper states that the residual was present in every checkpoint measured (SFT onward) and that its origin before
  SFT is undecided (Limitations).

## 2. Exploratory comparison (not confirmatory)
- Question: with the judgment readouts fixed to the two that are valid in every checkpoint (S, HARSH), how does the
  excess beyond alignment compare between the base models and the post-trained checkpoints?
- Data: base s1p main run (16 directions, 100 nulls, s1p_rating.py sha 7d687667, same as Stage 1s), started 11/16
  13:06 UTC by run_s1t_p_explore.sh. Post-trained: the existing Stage 1s (100 nulls) and Stage 1p (final models) s1p
  outputs.
- Code: p2_stage1t/s1t_explore.py. excess = s1s_excess.model_excess with vj = [S, HARSH] for **every** model. Stage 1s
  and 1r excess values (5-6 readouts) are **not** reused.
- Code check done before this registration (--check mode): the script reproduces the registered excess of all 11
  post-trained checkpoints with their registered readouts (max abs difference 9e-15). S+HARSH values were not computed.
- Families: Llama (base → Tulu-SFT → Tulu-DPO → Tulu final), OLMo (base → SFT → DPO → Instruct), Llama-Meta
  (base → Llama-3.1-8B-Instruct).
- Reported per model: mean excess over 16, n positive, one-sided t (uncorrected, descriptive), n_null, null-line slope
  and residual SD (scale diagnostics). Per adjacent stage: paired mean difference, two-sided t (descriptive).
- Outputs go to a new directory (results/p2_stage1t_explore); no existing output file is overwritten.

## 3. How results may be read (fixed now)
| base S+HARSH result | reading |
|---|---|
| mean > 0, one-sided p < .05 | "With rating-format readouts only, the base model also shows a positive excess." Consistent with, but not evidence for, a pretraining origin. |
| otherwise | "With rating-format readouts only, no positive excess was seen in the base model." Consistent with a post-training origin **or** with lower base sensitivity; the two cannot be separated. |
| base > SFT (paired) | reported as is: "the rating-format excess does not increase with post-training"; read together with the null-line diagnostics. |

- Results are reported whatever their direction.
- Placement: appendix only. The main-text claim and the Limitations text in §1 stay the same whatever the result.

## 4. Claims that are not allowed
- "The residual exists in the base model / originates in pretraining" or "arises in post-training" (confirmatory).
- Any change to the registered reading "not measurable".
- Comparing magnitudes across models without the scale diagnostics (n_null, null slope, residual SD).
- Reason: S and HARSH are both rating-format readouts. With no choice-format readout there is no lexical or format
  control, and the base models are less sensitive overall.

## 5. Timing
- This file's commit time is recorded by git. The end time of the base s1p job is recorded in
  /root/aiemo/logs/s1t_p_explore.log ("S1T_P_DONE"). The job was still running when this was committed.
