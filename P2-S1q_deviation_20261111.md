# P2 Stage 1q: deviation note (2026-11-11, written after the registered tests were run)

## H-val2: the registered tail has the wrong sign
- zJ_B is positive when judgments become harsher (the sign of L is reversed).
- In Stage 1p, H-val was the coefficient of the val rating (higher = more pleasant). A negative coefficient meant
  "more unpleasant, harsher".
- In Stage 1q, H-val2 was redefined as a group difference: mean(unpleasant) - mean(pleasant) of zJ_B. Under this
  definition, "more unpleasant, harsher" is a POSITIVE value. The registration nevertheless kept the Stage 1p sign
  and registered a one-sided LOWER test. The reading of the outcomes ("H-val2 significant (negative):
  unpleasantness makes judgments harsher") contains the same error.
- What we report as confirmatory: the test as registered. obs = +3.82, one-sided lower p = 1.0. H-val2 is not
  supported as registered.
- What we report as exploratory only: the upper-tail p (< .0001; all six models positive), and the same gap after
  also regressing on the projection z (0.15, p = .32).
- H-align (primary) is unaffected: obs = 1.46, p < .0001 (upper, as registered).

Exploratory analyses: s1q_explore.py (seed 9160), not registered.
