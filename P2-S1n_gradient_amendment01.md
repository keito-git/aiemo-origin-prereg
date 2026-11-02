# P2 Stage 1n gradient decomposition, amendment 01 (before any planned-model gradient run)

Registered 2026-11-02, after the smoke and before any run of s1n_grad.py on a planned model. No steered value of a
Stage 1n main direction has been looked at (Stage 1n: 4 of 6 models done, results copied but not opened).
Inputs: the smoke on Qwen3-8B-Base (unplanned), a finite-difference diagnostic on the same model, and external
review 2 (11/02).

## 1. Smoke criterion (changed after the smoke failed)
- Registered criterion: median relative error of central finite differences (+-alpha/4) vs the gradient < 5%.
  Smoke: 13.6% (21 rows: 3 random directions x 7 questions). FAILED; the queue stopped as designed.
- Diagnostic (s1n_grad_fddiag.py, Qwen3-8B-Base only): step alpha/16, /8, /4, /2.
  - Random directions: RMS error / RMS gradient 0.53, 0.28, 0.17, 0.16; corr(fd, grad) .895, .959, .986, .988.
    The error falls as the step grows: bf16 rounding noise in the finite differences dominates. The relative error
    is ill-defined for rows whose projection is near zero (these rows produced the failure).
  - Along the unit S gradient: fd / grad = 0.95, 0.90, 0.81, 0.60. The ratio approaches 1 as the step shrinks:
    the gradient is correct, and the curvature along steep directions is large at alpha.
- New criterion: corr(fd, grad) over random directions at alpha/4 >= .95, and fd / grad along the unit S gradient at
  alpha/16 within [0.9, 1.1]. The smoke is rerun with this criterion before the planned models.
- Consequence for the reading: the first-order prediction can be poor at alpha along steep directions. This is what
  the registered G0 gate (r >= .5 on the measured nulls) checks; G0 is unchanged.

## 2. Three-component split (new, secondary; external review 2)
The G1 residual cannot separate a judgment component from the meaning of the sanction words. With three judgment
readouts on the same texts (S, HARSH, and L reversed), the residual can be split further.
- r_i = s_i (g_i - P_U g_i) for i in {S, HARSH, L}, s = (+1, +1, -1); P_U = projection on span G_U (main:
  U_bright, U_fam; sensitivity: plus U_size).
- common c = mean of r_i (the part shared by the three sign-aligned judgment readouts beyond the degree controls);
  encoding-specific e_S = r_S - c; scalar part of S = P_U g_S.
- A_common(v) = alpha v . c; A_scalarS(v) = alpha v . P_U g_S; A_encS(v) = alpha v . e_S.
- K1 (secondary, one-sided): sum over G0-gated models of z_common(int_neg), standardized by the pair_neg null; null as
  in G1. K2 (secondary, two-sided): sum of z_common(int_neg) - z_common(int_neu), null as in G2.
- Descriptive: norms of c relative to g_S, cos between the residuals r_i, A_scalarS and A_encS for each direction.
- Limit (stated in advance): all three readouts are 1-9 digit ratings, so "encoding-specific" here means differences
  within the digit format only. L uses different words ("lenient") and its digit scale runs against the sanction, so
  c excludes a plain "larger digit" component; it can still contain meaning shared by the three wordings. The split
  across formats (multiple choice, pairwise) is planned for Stage 1o.
- G1 stays the primary test of the gradient analysis.

## 3. Files
p2_stage1n/grad/: s1n_grad.py (A_common, A_scalarS, A_encS; new smoke checks), s1n_grad_pooled.py (K1/K2),
s1n_grad_fddiag.py, queue_s1n_grad.sh, SHA256SUMS.
