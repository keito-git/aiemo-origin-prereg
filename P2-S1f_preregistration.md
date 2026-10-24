# P2 Stage 1f preregistration: negative valence, any evaluation, or rare words?

Registered before any Stage 1f run (no smoke run on the planned conditions). Follows Stage 1d (aa6523c), whose
registered result was: the word direction beats the situation direction on B2 (sum over 3 models, p = 0.004), and
anger is not distinguishable from fear or from emotion-free negative evaluation (all Holm p >= 0.23).
Internal reviews (R2, R3) named two alternative explanations for the word effect: (1) rare/surprising words,
(2) valence. This stage tests both.

## Stimuli (p2_stage1f/stage1f_closings.json)
- poseval_L: positive moral evaluation without emotion words, mirroring eval_L sentence by sentence.
- rare_L: low-frequency, non-evaluative, non-emotional words (quotidian, gazebo, vellum, funicular, ...).
- Every sentence k has the word count of neutral_C[k]; 0 hits against the emotion word list E.
- Manipulation check: mean closing surprisal of rare_L >= that of eval_L in a model; if not, F_rare is reported
  as uninformative for that model (it stays in the sum).

## Design
Anger situations of Stage 1c, models and revisions as in Stage 1d, layer round(0.33 x layers), alpha = 0.2 x mean
residual norm. The eigenbasis for the variance-matched nulls is built from the Stage 1d texts only.
Directions on B2: word_eval (re-run with the 1d seed), word_poseval, word_rare, sit_scaled (d_sit with alpha x
sd(word_anger)/sd(sit)). 1000 null draws each, seeds 7204, 7401, 7402, 7403.
Gate: Stage 1d pos_B2 (anger) p < 0.05.

## Registered tests (sum over models, paired null index i with i)
Primary, Holm over two:
- F_val = sum [Delta(word_eval) - Delta(word_poseval)], one-sided upper.
- F_rare = sum Delta(word_rare), one-sided upper.
Secondary, two-sided, no correction: F_pos = sum Delta(word_poseval); F_sit = sum Delta(sit_scaled).

## Interpretation table
| F_val | F_rare | F_pos | reading |
|---|---|---|---|
| sig | n.s. | sig negative | a bipolar valence axis drives B2 |
| sig | n.s. | n.s. or positive | negative evaluation drives B2; positive evaluation does not reverse it |
| n.s. | n.s. | sig positive | any evaluative word (salience of evaluation), not valence and not rarity |
| any | sig | any | rare words alone move B2; the word effect is at least partly a rarity effect |
| other | | | described case by case |
F_sit: if sig positive, the situation direction is weak only because its variance is small (Stage 1d "cannot say" #2).
