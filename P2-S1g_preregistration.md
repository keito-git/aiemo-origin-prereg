# P2 Stage 1g preregistration: confirmation in three new models, negated sanction readout, shared component

Registered before any planned Stage 1g run. A smoke run on an unplanned model (Qwen3-8B-Base) is listed in the
addendum if it is done. Inputs: Stage 1d (aa6523c) and Stage 1f (7412be7) results and the internal reviews of 10/25.

## Why
1. Stage 1d primary 1 (word direction > situation direction on B2) and Stage 1f F_val (negative evaluation >
   positive evaluation, raw p 0.030, Holm 0.060) were found in three models. Their null SD is the spread of the B2
   effect of random directions; it does not shrink with more closings or items. Power grows with the number of models.
2. R3: the word effect might only raise the probability of strict-sanction tokens. B2neg negates the frame
   ("rejected by", "does not call for") and sets the lenient option as target, so a stricter judgment raises Delta in
   both B2 and B2neg, while a token bias lowers Delta on B2neg.
3. R3: the anger, fear and eval directions (cos 0.5-0.8) may act through one shared component.

## Models
- New (independent of all earlier stages): mistralai/Mistral-7B-Instruct-v0.3 (c170c708c41dac9275d15a8fff4eca08d52bab71),
  ibm-granite/granite-3.1-8b-instruct (4009206d5fc95d2e65a7b7633e159d6e97e25d35),
  allenai/Llama-3.1-Tulu-3-8B (666943798adbde0b1aff34626007e26986a3c107). All ungated.
- Old: Qwen3-8B, llm-jp-4-8b-instruct, OLMo-2-1124-7B-Instruct (revisions as in Stage 1d).

## Runs (s1g_confirm.py, n_null 1000, layer round(0.33 x layers), alpha 0.2 x mean residual norm)
- New models: B2 dirs word_anger, sit, word_eval, word_poseval, word_joy, word_rare, pc1_neg, anger_res;
  B2neg dirs word_anger, word_eval, word_poseval, pc1_neg.
- Old models: B2 dirs pc1_neg, anger_res; B2neg dirs as above.
- pc1_neg = first principal axis of {word_anger, word_fear, word_eval}; anger_res = word_anger without pc1_neg.

## Gates
- B2: pos_B2 p < 0.05.
- B2neg: pos_B2neg p < 0.05 and item-wise corr(baseline B2, baseline B2neg) > 0 (the negation is understood).

## Registered tests (s1g_pooled_tests.py). Primary, Holm over three, one-sided upper
- G1 = sum_new [Delta_B2(word_anger) - Delta_B2(sit)]
- G2 = sum_new [Delta_B2(word_eval) - Delta_B2(word_poseval)]
- G3 = sum over gated models of Delta_B2neg(pc1_neg)
Secondary (no correction): G2 over all six models; sum_all6 Delta_B2(pc1_neg), Delta_B2(anger_res) (upper);
sum_new Delta_B2(word_rare) (upper); sum Delta_B2neg of word_anger / word_eval / word_poseval (two-sided).

## Reading
- G1 significant: the word > situation result replicates in new model families.
- G2 significant: negative evaluation, not evaluative salience in general, carries the effect.
- G3 significant: the effect follows the judgment, not sanction tokens. G3 significantly negative (p_lower < 0.05):
  a token bias. If fewer than two models pass the B2neg gate, G3 is reported as not testable.
- pc1_neg significant and anger_res not: a shared negative component carries the effect (descriptive).
