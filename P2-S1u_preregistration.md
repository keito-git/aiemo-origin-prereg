# P2 Stage 1u preregistration: is the residual beyond alignment present in the base models when a fixed format prefix makes their readouts valid?

Date: 2026-11-18. Committed before any steering run with the prefix (only no-steering base scores, a cache-equivalence check with one random vector, and a plumbing smoke with tiny null counts whose effect values are not read).
Design: `02_設計/17_段階1u_回答形式の手本で土台を測れるようにする_設計_draft_20261118.md` (project folder; §8 lists the changes after the internal R2/R3 review).
Code: `p2_stage1u/` (sha256 in `sha256.txt`).

## 1. Background and lineage (reported in the manuscript as is)
- Stage 1t (P2-S1t, commit fa63a9d): both base models (Llama-3.1-8B, OLMo-2-1124-7B) failed the Stage 1o entry rule (no valid choice-format readout). The registered reading "not measurable" stands and is not replaced by this stage.
- Stage 1t exploratory (S + HARSH only): Llama base showed a positive excess; rating-format-only, so the rating-scale alternative could not be excluded.
- Stage 1u was designed after seeing these results. It is a separate measurement in a different, fixed readout format.

## 2. Format prefix
- Every readout prompt of `readout_R9o_main.csv` is preceded by eight demonstrations of the answer format (`s1u_prefix_pilot.prefix(8)`, "F8"): neutral factual topics only (no sanction, harm, norm or emotion content); 4-choice answers A-D once each, 2-choice answers A and B once each, digits 9 and 1.
- Selection (no steering; excess never computed): F0, F4 and F8 were scored with the registered validity function. Rule (written after the pilot, which saw validity only): among versions that let both bases enter, take the one with more valid judgment readouts; tie -> the longer one. F8 was taken (Llama base 6 vs 5, OLMo base 5 vs 4). Pilot values are reported in full (design §1).
- Stopping rule: no further prefix sizes are tried, whatever the outcome.
- Direction texts, manipulation-check prompts, contexts, layer, alpha and the covariance are unchanged.

## 3. Models (all with F8)
| role | model | revision |
|---|---|---|
| Llama base | meta-llama/Llama-3.1-8B | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b |
| Llama SFT | allenai/Llama-3.1-Tulu-3-8B-SFT | f2a0b46b0cfda21003c6141b1ff837b7e165524d |
| Llama, Meta post-training | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| OLMo base | allenai/OLMo-2-1124-7B | 7df9a82518afdecae4e8c026b27adccc8c1f0032 |
| OLMo SFT | allenai/OLMo-2-1124-7B-SFT | 1de02c0175118a9de5854aec80a1f970e701e928 |

## 4. Procedure (`queue_s1u.sh`)
- `s1u_run.py o` = registered `s1o_rating.py` with two patches only: (1) the F8 prefix on readout prompts; (2) `Readout.logp` reuses the key/value cache of the prefix shared by all items (rebuilt for every steering vector; steering is added at every position, prefix included, as in the registered code). Gradients use full sequences.
- s1o: 100 null pairs, 10 minor nulls (used only for the gradient positive control and description; minimum attainable p = 1/101). s1p (`s1u_run.py p`): 16 contrasts, 100 nulls (as Stage 1s).
- Then `s1s_validity.py` (Stage 1o criteria unchanged, entry rule unchanged), `s1s_excess.py` (unchanged), `s1u_tests.py`.
- Cache criterion: the cache-induced change of a random-direction effect is below 10% of the null SD on every readout, both bases. Measured before registration (design §8.2): at most about 6% (L); the full computation itself differs by up to 0.0032 between batch sizes on the choice readouts. The initial draft criterion (1%) was not met and was changed before registration; both are reported.

## 5. Hypotheses and tests (`s1u_tests.py`)
- H-format (read first): Tulu-3-8B-SFT, OLMo-2-1124-7B-SFT, Llama-3.1-8B-Instruct; one-sided t of mean excess > 0 (df 15), Holm over three. The F8 format measures the residual only if at least two are positive.
- H-base (primary): each base, one-sided t (df 15), Holm over the two; "positive" = Holm p < .05. Secondary TOST (+-0.5); "zero" only if TOST p < .05. A base failing the entry rule is "not measurable" and leaves the family.
- Family check: a base is read only if the SFT checkpoint of its own family is positive with F8; otherwise its numbers are reported without an origin claim for that family.
- H-shift (secondary): per family, paired t over the 16 directions of (SFT - base), two-sided, Holm over two.
- Format scale (descriptive): F8 vs no-prefix (Stage 1s) excess for the three post-trained checkpoints; ratio of means outside [0.5, 2] is flagged, and then no magnitude is compared across formats.

## 6. Reading (fixed)
| base (H-format holds, family check passes) | reading |
|---|---|
| positive | residual present before post-training and kept through post-training (within the F8 format) |
| not positive, TOST zero | residual arises in post-training (by SFT) |
| not positive, undecided | suggests the residual arises in post-training |
| not measurable | origin not determined (as Stage 1t) |
- H-format fails -> no origin claim from this stage.
- Positive base and significant H-shift with SFT > base -> "present in the base and strengthened at SFT"; not significant -> "present in the base, not clearly changed at SFT" (with the scale caveat).
- H-data (Tulu SFT-data ablation) is not run in this stage.

## 7. Exploratory (reported, not confirmatory)
- F8 vs no-prefix excess per direction for the post-trained checkpoints; category (7) vs magnitude (9) directions; leave-one-readout-out with L-out and choice-format-only excess shown explicitly; null-line slope and residual SD; direction cosines base vs SFT.
- Wrong-answer prefix (F8 with every demonstrated answer wrong), no steering, validity of the two bases (`pilot_s1u_wrong`; already run before this commit): neither base enters (choice-format severity effects 0.33-0.45, back to the F0 level). The prefix works by showing correct task performance, not the answer format alone.

## 8. Alternatives stated in advance
- Rating-scale use (Stage 1t weakness): addressed by choice-format readouts and the choice-only excess.
- Steering at the prefix positions may affect the emotion direction and random directions asymmetrically; the prefix contains no sanction or emotion words; stated as a residual limitation.
- The prefix changes how the task is performed (F8 raised no-steering severity effects 1.5-3.5x, also in post-trained models; the wrong-answer prefix does not make the bases valid): the base result is a property of the base model led to perform the task by in-context demonstrations (F8), not of the base model without a prefix. The manuscript states this scope.
