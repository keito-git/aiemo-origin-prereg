# P2 Stage 1g amendment 01: B2neg manipulation check (before any planned run)

Written after the smoke run on an unplanned model (Qwen/Qwen3-8B-Base, rev 49e3418f, n_null 2) and before any of
the six planned runs. No planned-model output exists. Directions, seeds, nulls, readouts B2/B2neg and the three
primary tests are unchanged.

## What the smoke run showed
- Code ran end to end (exit 0, 520 s); variance checks 1.000; pos_B2 and pos_B2neg p = 0.005 (200 nulls).
- Raw manipulation check corr(baseline B2, baseline B2neg) = -0.86.

## Why the registered check is not valid
Each item score is option prior + judgment. B2 targets the strict option and B2neg the lenient one, so the
option prior P enters B2 as +P and B2neg as -P. If the negation is understood, B2 = P + J and B2neg = -P + J; if it is
ignored, B2neg = -B2. The raw correlation is negative in the second case and can also be negative in the first case
whenever var(P) > var(J). It therefore cannot tell whether the negation is understood.

## Change
1. New readout B2neu (readout_B2neu.csv, build_b2neu.py): the same 100 items with a stance-free frame
   ("The response mentioned in the report was", "According to the written policy, this case is about"),
   target = strict option. Scored once without steering; it estimates P per item.
2. B2neg gate: pos_B2neg p < 0.05 and corr_adj = corr(B2 - B2neu, B2neg + B2neu) > 0 over items.
   If the negation is ignored, corr_adj is about -1; if it is understood, it is positive when the prior estimate
   is good. The raw correlation is still reported (descriptive).
3. s1g_confirm.py also saves per-item baselines (B2, B2neg, B2neu). No other change.
4. s1g_pooled_tests.py: the prereg already says G3 is not testable if fewer than two models pass the B2neg gate.
   The code now implements this; G3 then enters the Holm correction with p = 1 (Holm over three stays).

The Qwen3-8B-Base smoke run with the amended code is reported in the addendum below before the planned runs start.

## Addendum: amended smoke run
- Run: Qwen3-8B-Base, amended code, dirs pc1_neg only, n_null 2; exit 0.
- Raw corr -0.865; corr_adj -0.684. Item SD: B2 4.05, B2neg 4.49, B2neu 3.32; corr(B2, B2neu) 0.74,
  corr(B2neg, B2neu) -0.76. The prior is large and shared, and after removing it the base model still treats
  "rejected" like "chosen": it fails the B2neg gate. This is expected for a base model and is not a planned result.
- The code paths run; the planned runs start after this commit.
