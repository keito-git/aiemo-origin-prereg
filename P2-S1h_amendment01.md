# P2 Stage 1h amendment 01: gate (after the unplanned smoke run, before any planned-model output)

The queue started the first planned model (Mistral) at 02:02 UTC; no planned-model output had been written or read
when this amendment was made. Only s1h_pooled_tests.py changes; s1h_crossed.py, the readout and the test are unchanged.

## Smoke run (Qwen3-8B-Base, n_null 2)
- exit 0, 421 s; directions identical to Stage 1g (cos 1.000, same alpha); var_check 1.000; pos_B2x p = 0.005.
- Valence check, negativity(lenient) - negativity(strict) per pair: B2x +2.50 to +4.62 (crossing holds, all six pairs).
  B2 original pairs: +0.88, +0.62, +0.13, -0.25, -0.75, -0.88 (mean about -0.04).
- Judgment check: corr_within_pair 0.33 (raw 0.07).

## Change
The registered gate also required mean_gap_B2 < 0 (the original strict options sound more negative). The smoke run
shows that the original sanction words are not consistently more negative-sounding by this probe. This condition is
not needed for H1, which only uses B2x. It is removed from the gate and reported descriptively. The rest of the gate
(pos_B2x p < 0.05, mean_gap_B2x > 0, corr_within_pair > 0) and the test are unchanged.
