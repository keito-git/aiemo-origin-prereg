# P2 Stage 1h preregistration: stricter judgment vs bias toward negative-sounding tokens

Registered before any planned Stage 1h run. A smoke run on an unplanned model (Qwen3-8B-Base, n_null 2) starts
the queue automatically; its code-level result is added as an addendum. Inputs: Stage 1g results (30eecf4, 22b5dd4)
and the internal reviews of 10/26 (R2, R3: the sanction-token alternative must be tested without negation, because
all six models failed the Stage 1g negation check).

## Question
The shared negative component pc1_neg makes B2 stricter (Stage 1g, sum of six models, p = 0.001). Is this
(J) a stricter judgment, or (T) a higher probability of negative-sounding option tokens ("a fine", "a suspension")?

## Readout B2x (readout_B2x.csv, build_b2x.py)
Same 100 B2 items (scenario, frame, option position, target = strict option). Each of the six B2 option pairs is
replaced by one crossed pair: the strict option worded in neutral/positive terms, the lenient option in negative terms
(e.g. "full accountability" / "careless neglect"). (J) predicts Delta_B2x > 0; (T) predicts Delta_B2x < 0.

## Runs (s1h_crossed.py, n_null 1000)
Six models of Stage 1g (same revisions). Directions rebuilt as in Stage 1g and checked against the saved ones (cos).
Directions scored: pc1_neg, word_anger, word_eval, word_poseval. Seeds 7701-7705. Positive control on B2x
(odd -> even items, 200 nulls).

## Gate (per model)
pos_B2x p < 0.05; valence check: mean over pairs of negativity(lenient) - negativity(strict) > 0 for B2x and < 0 for
B2 (negativity = log-odds of " negative" vs " positive" after 'The phrase "<option>" sounds', no steering);
judgment check: corr(B2, B2x baselines) over items after removing each option pair's mean > 0.
Fewer than two models passing: H1 not testable.

## Registered test (s1h_pooled_tests.py)
H1 = sum over gated models of Delta_B2x(pc1_neg). p_upper < 0.05: supports (J). p_lower < 0.05: supports (T).
Neither: undecided. Secondary (two-sided, no correction): the same sum for word_anger, word_eval, word_poseval;
sum [Delta_B2(pc1_neg) - Delta_B2x(pc1_neg)] (size of the wording effect; B2 values from Stage 1g).
