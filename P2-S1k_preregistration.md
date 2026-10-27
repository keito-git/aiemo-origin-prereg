# P2 Stage 1k preregistration: a control question on the same scenario text

Registered before any planned steered Stage 1k readout is run. Before registration:
- A baseline and positive-control probe was run on the six planned models (s1k_probe.py). No target direction was
  steered. The results are reported in full below and in p2_stage1k/s1k_probe_*.json.
- One development smoke of s1k_rating.py was run on the unplanned Qwen3-8B-Base with two nulls, to debug the code.
  Only code-level outputs were inspected: the run completed, cos with Stage 1g was 1.0, and var_check was 1.0. The
  readout checks were also inspected: S severity +2.32 (100% of pairs), C severity +0.33, C location rho 0.87. Its
  target-direction effects were not inspected before registration. That run is in results/smoke_s1k_dev and is not
  used in any test.
Inputs: Stage 1j results (10b4eea) and the internal reviews of 10/28 (R2, R3).

## Why
Stage 1j: K1 held (pc1_neg raises the 1-9 deserved-sanction rating; sum +0.94, Holm p 0.034, 4 gated models). K2 did
not hold (vs an object-weight rating, p 0.086), so a general bias toward larger ratings is not excluded. The object
control used different text. R3 asked for a control question on the same scenario text. R2 and the lead reviewer noted
two further problems:
- K1 depended on two models.
- The positive control (" 8" minus " 2") failed in two models.
- The pattern "larger effect in severe scenarios" could come from saturation near rating 1.
Stage 1k addresses these points with new scenarios, the same-text control, a saturation-free metric, and a second
positive control.

## Stimuli (build_r9k.py -> readout_R9k.csv)
- There are 60 new rule-violation scenarios, none used in earlier stages. Six drafts that repeated an earlier scenario were
  replaced. Two items still share a generic 4-word sequence with earlier stimuli ("an employee used the", "a
  warehouse worker stacked") but describe different acts.
- Location is designed: 30 scenarios are indoors and 30 outdoors.
- Each scenario gets the Stage 1j mild or severe added sentence.
- Two questions are asked about the same text, in qa format ("Scenario: ...\nQuestion: ...? Answer with one
  digit.\nAnswer:"):
  - S: "On a scale from 1 (no sanction at all) to 9 (the most severe sanction), how severe a sanction does this case
    deserve?"
  - C: "On a scale from 1 (certainly indoors) to 9 (certainly outdoors), where did this take place?"
- In total, 240 items: 60 scenarios x mild/severe x S/C.

## Pre-registration probe (s1k_probe.py; no target direction; 50 nulls on the context covariance)
| model | S severe-mild | S mild / severe | C severe-mild | C rho location | pos_82 dE_S (p) / dE_C | pos_sev dE_S (p) / dE_C |
|---|---|---|---|---|---|---|
| Mistral-7B | +4.63 | 2.14 / 6.77 | +0.31 | 0.86 | +0.38 (0.10) / +0.33 | +0.66 (0.02) / +0.71 |
| granite-3.1-8b | +5.64 | 1.98 / 7.63 | +0.29 | 0.75 | +0.87 (0.02) / +1.52 | -0.26 (0.67) / -0.30 |
| Tulu-3-8B | +5.20 | 2.20 / 7.40 | -0.13 | 0.87 | +0.54 (0.02) / +0.45 | -0.02 (0.51) / -0.06 |
| Qwen3-8B | +4.05 | 3.22 / 7.27 | +0.41 | 0.85 | +0.09 (0.31) / +0.45 | +0.79 (0.02) / +0.18 |
| llm-jp-4-8b | +3.38 | 3.45 / 6.83 | +0.15 | 0.87 | +0.35 (0.02) / +0.23 | +0.05 (0.25) / -0.37 |
| OLMo-2-7B | +4.72 | 2.50 / 7.22 | +0.27 | 0.86 | +0.78 (0.02) / +1.12 | -0.43 (0.92) / +0.14 |

- S tracks the severity manipulation in all models (100% of pairs).
- C tracks the designed location (rho 0.75-0.87) and moves little with severity (|C severe-mild| <= 0.41).
- C ratings sit near 1 or 9 (bimodal). S ratings are mid-scale. Mean E therefore saturates differently on S and on C.
  The slope metric below is used for the S - C contrast.
- pos_82 (a "say a larger digit" direction) moves C as much as, or more than, S. A general digit bias therefore
  shows up on C, and S - C can detect it.
- pos_82 fails on S in Mistral and Qwen3. pos_sev (severe minus mild hidden state) passes there and fails elsewhere.
  The gate accepts either.

## Readout and metrics (s1k_rating.py)
- Per item: E = expected digit from the renormalised next-token probabilities of the nine digits (as in Stage 1j).
- slope = OLS slope over k of [log p_steered(k) - log p_base(k)]. A uniform shift of the digit logits along k gives
  the same slope at any baseline position, so the metric does not saturate.
- Effects are means over items: dE_S, dE_C, dslope_S, dslope_C, and severe-minus-mild differences of dE_S and
  dslope_S.
- Directions, layer (round(0.33 L)), alpha (0.2 x mean context norm) and the null covariance are rebuilt exactly as
  in Stage 1j, with a cos check against Stage 1g.
- Directions: pc1_neg, word_anger, word_eval, word_poseval, val_eval = unit(word_eval - word_poseval).
- Nulls are variance-matched random directions. All six columns come from the same random direction (index-paired).
  n = 1000 for pc1_neg and 500 for the others. Seeds 8201-8205.
- Positive controls: the direction is built on odd scenarios and measured on even S items, with 200 nulls each.
  pos_82 uses seed 8206 and pos_sev seed 8207.

## Runs
Six models with the Stage 1g revisions. Queue (queue_s1k.sh):
1. Smoke on Qwen3-8B-Base with n_null 2.
2. Automatic check of code and of S and C readout validity. The queue stops if this check fails.
3. Run the six models.
4. Run the tests (s1k_pooled_tests.py).

## Gate (per model)
- pos_82 or pos_sev p < 0.05.
- S severity effect > 1, positive in >= 80% of pairs.
- C rho with location > 0.5 and |C severity effect| < 1.
- If fewer than two models pass, the primary tests are not testable.

## Registered tests (s1k_pooled_tests.py)
Primary, one-sided upper, Holm over two:
- K1k = sum over gated models of dE_S(pc1_neg). This replicates Stage 1j K1 on new scenarios.
- K3 = sum [dslope_S - dslope_C](pc1_neg), with paired nulls. The sanction rating rises more than the location
  rating of the same text, on the saturation-free metric.

Secondary (no correction):
- K3e = sum [dE_S - dE_C](pc1_neg), upper.
- sum dslope_S(pc1_neg), upper.
- sum dE_C(pc1_neg) and sum dslope_C(pc1_neg), two-sided.
- Severity interaction: sum of severe-minus-mild dslope_S(pc1_neg) and of severe-minus-mild dE_S(pc1_neg),
  two-sided.
- sum dE_S for word_eval, word_anger, val_eval (two-sided) and for word_poseval (upper).
- sum [dslope_S - dslope_C] for word_eval, word_anger, val_eval, word_poseval, two-sided.
- sum [dE_S(word_eval) - dE_S(word_poseval)], two-sided.

Descriptive: per-model effects, and pos_82 on S and C as a benchmark of a general digit bias.

## Reading
- K1k and K3 significant: negative-evaluation steering raises the sanction judgment specifically. The rise exceeds the
  same model's rating of an unrelated property of the same text, so a general digit bias does not explain it.
- K1k significant, K3 not: the stricter rating replicates on new scenarios, but a general digit bias is not excluded.
- K1k not significant: the Stage 1j K1 result does not replicate on new scenarios.
- Interaction (slope): a positive value means the effect grows with case severity even without saturation. Zero means
  the Stage 1j severe/mild pattern was a saturation artefact.
- word_poseval: no effect is expected (Stage 1i B2n, Stage 1j).
