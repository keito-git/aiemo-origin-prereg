# P2 Stage 1n gradient decomposition: is the sanction slope change carried by a sanction-specific gradient?

Registered while the Stage 1n queue (f401d8d) is still running (4 of 6 models done, 2026-11-02 00:30 UTC).
No steered value of a Stage 1n main direction has been looked at. What has been seen before registration:
- For Mistral, granite, Tulu and Qwen3: the positive-control p values (pos_82, pos_sev, pos_U82), the no-steering
  checks (U_bright validity, severity effects) and, for Mistral, cos(pos_82, pos_U82).
- Not seen: any effect of int_neg, int_neu, int_pos or pc1_neg, and any null distribution of these directions.
  The result files were copied with md5 checks but not opened.
Inputs: external review 1 (10/31, Jacobian proposal), internal review R3 of this plan (10/31), Gao, Peng, Wakamiya,
Aramaki, arXiv:2608.22985 (sections 3.4, 5.1, appendix B.6 read directly on 11/02).

## Why
A first-order prediction of a steering effect from the gradient (as in Gao et al., r = .83 to .96) does not separate
a degree projection from a judgment change: both paths are in the gradient. What can separate them is where the
sanction gradient lies relative to the gradients of degree questions that are irrelevant to the judgment.
Gao et al. decompose the direction v on an SVD basis of answer-identifier gradients at the pre-answer position. Here
the sanction gradient is decomposed on the span of the gradients of control degree questions (U_bright, U_fam).

## Quantities (s1n_grad.py)
- For each Stage 1n question t, g_t = gradient at delta = 0 of the item mean of w . log p(k), k = 1..9,
  w = (k - 5) / sum (k - 5)^2, with delta added at every position of the output of decoder layer L-1 (the Stage 1n
  steering site). Per-item gradients are taken with one delta per item; bf16 forward, fp32 delta. alpha v . g_t is
  the first-order prediction of dslope_t(v). Items, layer, alpha and model revisions as in Stage 1n.
- Main split: G_U = [g_U_bright, g_U_fam]; g_S^shared = projection of g_S on span G_U; g_S^res = g_S - g_S^shared.
  Sensitivity split: G_U plus g_U_size (closer in meaning to the direction; may remove sanction-specific parts).
- A_res(v) = alpha v . g_S^res; A_shared(v) = alpha v . (g_S^shared - g_U_bright);
  Dhat(v) = A_res + A_shared = alpha v . (g_S - g_U_bright).
- Null directions: the Stage 1n nulls rebuilt from their seeds (pair 8501, int_pos 8502, pc1_neg 8503). The run checks
  cos with the saved Stage 1n directions, alpha, and the saved pair cosines.

## Registered tests (s1n_grad_pooled.py)
G0 (gate, per model): Pearson r between Dhat and the measured D_S,U_bright over the 600 pair_neg nulls of Stage 1n.
Models with r >= .5 enter G1 and G2. Sensitivity: gates .3 and .7 (reported, not used for the decision).

G1 (primary, one-sided, alpha .05): T = sum over gated models of z_res(int_neg), where z_res standardizes A_res by the
model's pair_neg null (mean, SD). Null: the same sum over the null index j. p = (#{T_j >= T} + 1) / (601).
Requires at least two gated models; if only one, its p is doubled.

G2 (secondary, two-sided): sum of z_res(int_neg) - z_res(int_neu) (int_neu standardized by the pair_neu null);
null: sum of z(r_neg_j) - z(r_neu_j).

Descriptive (not tested): res norm fraction of g_S; cos between question gradients; per-item cos of g_S,i with span
G_U; |A_shared| and |A_res| per direction; measured dslope at alpha and 2 alpha for the four directions; the
sensitivity split.

## Reading
- G1 and N1 significant: the sanction change is carried by a sanction gradient not shared with the control degree
  questions. A projection on the degree readout shared with them (C') does not explain it.
- G1 not significant, N1 significant: N1 can be explained by a different sensitivity of S and U_bright to a shared
  degree axis. This is compatible with the projection account.
- N1 not significant: G1 is descriptive only.
- What G1 cannot separate (stated in advance): a projection on the meaning of the sanction words themselves (C'',
  e.g. "ban" heavier than "no action"), which is specific to S and so lies in g_S^res; and nonlinear paths. G1 is
  read together with the Stage 1o categorical readout and the int_neg vs int_neu comparison, not alone.

## Runs
1. Smoke on an unplanned model (Qwen3-8B-Base, empty res1n) with --fd_check 3: the gradient must match central finite
   differences (+-alpha/4) with median relative error < 5%. Bug fixes found here are recorded as an amendment with the
   diff before any planned model runs.
2. After the Stage 1n queue ends: s1n_grad.py on the six models (one at a time, waiting for free GPU memory; other
   jobs on the GPU are not touched), then s1n_grad_pooled.py.
