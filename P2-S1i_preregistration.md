# P2 Stage 1i preregistration: the wording-independent part of the steering effect

Registered before any Stage 1i judgment readout is run (planned or smoke). Only a valence probe of candidate option
phrases (no steering, no judgment readout) was run to choose the option pairs. Inputs: Stage 1h results (05f5037,
0550bc2) and the internal reviews of 10/26.

## Why
Stage 1h: on B2 the steering effect is the sum of (a) a valence-congruent output bias (directions raise options that
sound like them) and (b) a part that does not follow wording (exploratory decomposition). R2: the decomposition rests
on equal readout sensitivity, which fails (B2x positive control about 26% smaller). R3: (b) may be a bias toward
formal/normative words ("full accountability"), not a stricter judgment. A readout whose options differ only in the
size of the sanction holds valence and formality fixed and measures (b) directly.

## Readouts
- B2m (primary; readout_B2m.csv, build_b2m.py): same 100 B2 items; both options use the same words and differ only in
  size ("a fee of 200 dollars" / "a fee of 2 dollars", "a two-month ban" / "a two-hour ban", ...). Target = larger.
  Valence probe over the six planned models: mean gap per pair -0.36 to +0.08.
- B2n (secondary; readout_B2n.csv, build_b2n.py): options differ in formality ("a formal response" / "an informal
  response", "a hearing" / "a chat", ...). Target = the formal option. Valence probe mean gap per pair -0.83 to +0.65.

## Runs
s1i_magnitude.py (B2m) and s1i_neutral.py (B2n), n_null 1000, six Stage 1g models and revisions, directions rebuilt as
in Stage 1g (cos check), dirs pc1_neg, word_anger, word_eval, word_poseval. Seeds B2m 7901-7905, B2n 7801-7805.
Queue: smoke on Qwen3-8B-Base (n_null 2) -> code-level check -> B2m six models -> B2m tests -> B2n six models -> B2n tests.

## Gate (per model, each readout)
pos p < 0.05; |mean valence gap| < 1; corr_within_pair (pair-demeaned baseline corr with B2) > 0.

## Registered tests (s1im_pooled_tests.py), primary, Holm over two
- I1 = sum over gated models of Delta_B2m(pc1_neg), one-sided upper: a stricter judgment independent of wording.
- I2 = sum [Delta_B2m(word_eval) - Delta_B2m(word_poseval)], two-sided: does negative evaluation exceed positive
  evaluation once valence and formality are held fixed?
Secondary (no correction): sum Delta_B2m of word_eval, word_poseval (upper), word_anger (two-sided); the same sums
divided per model by that model's positive-control delta (sensitivity-normalised); the B2n versions of I1/I2
(s1i_pooled_tests.py); sum [Delta_B2n(pc1_neg) - Delta_B2m(pc1_neg)] two-sided (formality bias).

## Reading
- I1 significant: a wording-independent stricter judgment exists (the (b) path is not only wording).
- I1 not significant while B2n is positive and exceeds B2m: (b) is largely a bias toward formal/normative words.
- I2 non-significant: the Stage 1g G2 result (negative > positive evaluation) is carried by wording valence.
