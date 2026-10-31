# P2 Stage 1n preregistration: does the intensity effect move the judgment, or every question that asks for a degree?

Registered before any steered Stage 1n run of a planned model. Before registration:
- Two no-steering probes (s1n_probe.py; readout_R9n_probe.csv, readout_R9n_probe2.csv) were run on the six planned
  models. They report only baseline ratings of candidate questions. The results are in p2_stage1n/probe/ and are
  summarised below.
- A smoke run of s1n_rating.py on an unplanned model (Qwen3-8B-Base, tiny nulls) checked the code.
Inputs: Stage 1m results (p2_stage1m), internal reviews R2/R3 of the Stage 1m results and R3 of the Stage 1n design
(10/31).

## Why
Stage 1m: the anger intensity direction (int_neg) and a non-emotional intensity direction (int_neu, large vs
enormous) both raised the sanction rating S more than a location rating C of the same text (sum J_S(int_neg) p 0.001;
J_S(int_neu) p_two 0.032). The leniency rating L did not fall (M1 p 0.60). The strongest remaining alternative (R3,
C') is a degree projection: an intensity direction raises the answer to any question that asks "how much". C asks
indoors vs outdoors, which is not a degree question, so Stage 1m cannot reject C'.
- Strong form of C': every degree question rises about as much as S.
- Weak form of C': a degree question rises more when its wording is close in meaning to the direction.

## Readout (readout_R9n.csv, built by build_r9n.py)
The 60 Stage 1k scenarios x mild/severe, each with seven 1-9 questions (840 items). The rows are copied unchanged from
the files the probes used.
- S (deserved sanction), C (indoors/outdoors), L (leniency): the Stage 1m rows.
- HARSH: "how harshly should this case be treated" (a paraphrase of S).
- U_bright: "how brightly lit was the place where this happened" (1 very dim, 9 extremely bright).
- U_fam: "how familiar is this kind of place to most people".
- U_size: "how large was the place where this happened" (1 very small, 9 extremely large). Its wording is close to
  the neutral intensity pairs (large / enormous); it is the probe for the weak form of C'.
- E = expected digit; slope = OLS slope over k of [log p_steered(k) - log p_base(k)], as in Stages 1k-1m.
- D_X,Y(v) = dslope_X(v) - dslope_Y(v).

## Probes (no steering)
Validity rule for a degree question U (set after probe 1, tightened after the design review): |severity effect| <
1.2, Spearman rho(E mild, E severe) over scenarios > 0.5, and |rho(U, S)| < 0.3 over the 120 texts.

| question | passes | note |
|---|---|---|
| U_bright | 6/6 | severity effect -0.98 to +0.21; rho(mild, severe) 0.55 to 0.80; rho(S) -0.28 to +0.23 |
| U_fam | 5/6 | Mistral rho(S) -0.40 |
| U_formal | 4/6 | not used |
| U_size | 3/6 | follows severity in Tulu, Qwen3, OLMo; used only for the gradient |
| U_unusual, U_crowd, U_noise, U_old | 2/6 or leaks in OLMo / llm-jp | not used |
| MILD ("how mild a response") | rises with severity in 6/6 | not used: the models answer it as an amount of response |

S, C and L reproduced the Stage 1m baseline. The only difference (items 350-359, up to 0.31 in Qwen3) comes from the
batch shape in bf16: these ten items formed the last partial batch in Stage 1m. With the same batch shape the
difference is 0.000 in all six models. The Stage 1m effects are not affected, because the baseline and the steered
runs use the same batches.

## Directions
int_neg, int_neu, int_pos and pc1_neg, built as in Stage 1m (the run checks cos with the saved Stage 1m directions).
Layer, alpha and the null covariance as in Stages 1k-1m. int_com, int_dif and pc1_pos are dropped (cost).

## Nulls (variance-matched random directions, as in Stage 1m)
- Paired null for int_neg and int_neu, n = 600 (seed 8501), cos matched to cos(int_neg, int_neu), redrawn if off by
  more than 0.01. n is 600, not 1000, because the readout has seven questions instead of three (cost about 8 h per
  model).
- int_pos and pc1_neg: 300 independent draws each (seeds 8502, 8503).
- Positive controls, 200 draws each: pos_82 and pos_sev on S (as in Stage 1m; seeds 8504, 8505) and new pos_U82 on
  U_bright (seed 8506): the mean hidden state of the U_bright prompts + " 8" minus + " 2" on the odd scenarios, read
  out on the even scenarios.

## Gate (per model)
- As in Stage 1m: pos_82 or pos_sev p < 0.05; S severity effect > 1, positive in >= 80% of pairs; C rho with location
  > 0.5 and |C severity effect| < 1; L severity effect < -0.5, negative in >= 80%, rho(S, L) < -0.5; valence check > 2.
- New: U_bright validity (rule above, recomputed in the run) and pos_U82 p < 0.05.
- Anger checks and neutral checks as in Stage 1m (manipulation checks recomputed in the run).
- N1 uses the models that pass the gate and the anger checks. A test needs at least two models; if only one is
  testable, its p is doubled.

## Registered tests (s1n_pooled_tests.py)
Primary (one test, one-sided, alpha 0.05):
- N1 = sum of D_S,U_bright(int_neg), upper tail; null = the r_neg half of the paired null. The anger intensity
  direction raises the sanction rating more than a degree question about the place.

Secondary (no correction):
- D_S,U_fam(int_neg) (models where U_fam is valid) and D_S,U_size(int_neg), upper.
- Weak C' gradient: D_U_size,U_bright(int_neu), upper (neutral checks); D_U_size,U_bright(int_neg), two-sided.
- D_S,U_bright(int_neu), upper (neutral checks); D_S,U_bright(int_neg) - D_S,U_bright(int_neu), two-sided, paired null.
- N2 = D_L,U_bright(int_neg), lower; D_L,U_bright(int_neu), two-sided.
- D_HARSH,U_bright(int_neg), upper.
- D_S,C(int_neg), upper (Stage 1m J_S again; same items, not independent).
- D_U_bright,C(int_neg), D_U_bright,C(int_neu), D_U_size,C(int_neu), two-sided: does a degree question itself move.
- D_S,U_bright of int_pos (joy checks) and pc1_neg, two-sided.
- Descriptive: summed dslope per question minus the summed null mean.

## Reading
- N1 significant: the sanction rating rises beyond a degree question about the same text. The strong form of C' does
  not explain the intensity effect.
- N1 not significant and D_U_bright,C(int_neg) close to D_S,C(int_neg): intensity raises degree answers in general
  (strong C'). The claim becomes a rating bias, not a judgment change.
- Gradient D_U_size,U_bright(int_neu) significant: the weak form of C' holds for the neutral direction; then a
  significant N1 rejects only the strong form, and the int_neg vs int_neu comparison must be read with this in mind.
- The anger > neutral difference found exploratorily in Stage 1m (harshness index H) is not tested here. It moves to
  Stage 1o with new scenarios, because N1 decides how H is read.

## Runs
1. queue_s1n_smoke.sh (before registration): smoke on Qwen3-8B-Base with tiny nulls and an automatic code and readout
   check.
2. queue_s1n.sh (after registration): the six models (Stage 1g revisions). After each model the queue prints the
   pair-cosine check. Then s1n_pooled_tests.py.
