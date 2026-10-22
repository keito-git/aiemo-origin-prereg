# P1-1 Preregistration: Is the shared non-human, non-lexical structure specific to emotion?

- Date: 2026-10-22. Committed before any P1-1 activation is extracted.
- Follows P1-0 (GO) and internal reviews (R2 Borderline, R3 Major surgery). Design notes: Project 1 `02_設計/01_P1-1設計_草案_20261022.md`.
- Main question: P1-0 found that model RDMs over emotion words contain structure that is explained neither by human VAD nor by GloVe, and that this residual is strongly shared across model lineages. **Is this specific to emotion, or a general property of how LLMs organise adjectives?**

## 1. Stimuli
- File `code/stimuli/p1_1_stimuli.csv`, SHA-256 `c3dfeaa70ee9b484e918cfb6f60814e6d3634701266f0bdaaef5ca122a11d3d1`.
- Two domains in identical frames (5 frames: "She was {w}.", "He looked {w}.", "They seemed {w} that day.", "I was {w} at that moment.", "The man became {w}.").
- Emotion: 171 words (Paper 1). Non-emotion: 58 person-describing adjectives (temperature, size, body, speed, age, health, wakefulness, etc.); `bitter` and `sleepy` belong to the emotion list and are not used as non-emotion items.

## 2. Models (12, eight lineages; revisions frozen as in Paper 1 registry v4)
A Qwen3-8B-Base, Qwen3-8B; B BAR-7B; C Apertus-v1.1-4B, Apertus-v1.1-4B-Instruct; D granite-4.2-8b; E llm-jp-4-8b-base, llm-jp-4-8b-instruct; F gpt-oss-20b; G GLM-4.7-Flash; H gemma-4-12B-it; supplementary (excluded from decision statistics): Mistral-Small-3.2-24B-Instruct-2506.
A model that fails to load in either transformers 4.57 or 5.17 is reported and dropped; the decision requires at least 6 lineages.

## 3. Representations
- Raw text, default special tokens, bf16. Positions: `word` (mean over target-word tokens; **primary**) and `last` (sentence-final token; sensitivity).
- All layers (embedding output + every decoder layer). Paper-1 preprocessing (column-centre, column-SD, 5 SD clip) applied within each domain; word centroid over frames; cosine RDM. Split-half RDMs: frames {1,3,5} vs {2,4}.

## 4. Predictors
- Human H: (i) Euclidean distance in NRC-VAD (V, A, D); (ii) Euclidean distance in the 11 Lancaster sensorimotor dimensions (z-scored). Both domains use the same two predictors.
- Lexical L: cosine distance in (i) GloVe 6B-100d, (ii) GloVe 840B-300d, (iii) fastText crawl-300d-2M-subword. Multiword items = mean of token vectors.
- Items: words with NRC-VAD and Lancaster values and at least one token in every embedding (expected: emotion ~149, non-emotion ~52).

## 5. Measures (per layer, per domain)
- Residual `e_m` = rank RDM residualised (OLS on ranks) on H(i), H(ii), L(i), L(ii), L(iii).
- **S_ind**: mean Spearman correlation of residuals between individual models of different lineages (main models).
- **U_matched** = S_ind / mean residual correlation between individual models of the same lineage.
- Variance of the model RDM explained by H, by L, by H+L (rank R²; medians).
- Antonym proximity: WordNet antonym pairs within each domain (same extraction rule as P1-0 D3); percentile of their mean residual.
- **Size matching**: the emotion domain is randomly subsampled to the non-emotion item count, 1,000 times (seed 20261022); emotion statistics are averaged over subsamples.

## 6. Primary statistic and decision rule
- Layer band: relative depth 0.25–0.75 (layers indexed 0..L with 0 = embeddings; relative depth = layer/L).
- **Δ = mean over the band of (S_ind,emotion − S_ind,non-emotion)**, word position, size-matched.
- Uncertainty: 1,000 item bootstrap resamples within each domain (resampling words, recomputing RDM subsets and residuals); 95% percentile CI.
- **EMOTION-SPECIFIC**: Δ ≥ 0.05 and the 95% CI excludes 0.
- **GENERAL** (the shared residual is a property of LLM adjective organisation, not of emotion): CI includes 0, or Δ < 0.05, or Δ < 0 (if Δ < 0 with CI excluding 0, report as "non-emotion more shared").
- Both outcomes are reported as results. Under GENERAL, P1 claims are reframed to LLM-general lexical organisation, and CAI-specific claims are deferred to functional tests (Projects 2–3).
- Secondary (not gates): U_matched per domain; antonym percentile per domain and its emotion − non-emotion difference; human- and lexical-explained variance per domain; `last` position; layer curves.

## 7. Quality rule (fixed in advance)
A model whose median-over-layers Spearman between its emotion RDM and H(i) is below 0.10 is excluded from the main statistics and reported separately (responds to the Apertus-1.5B anomaly in P1-0).

## 8. Positive and negative controls (run before the real data)
Synthetic RDMs = H + L + s·Z_shared + model-specific noise, with s ∈ {0, 0.25, 0.5, 1}. The pipeline must return S_ind ≈ 0 for s = 0 (|S_ind| < 0.05) and increase monotonically with s. If not, the pipeline is fixed before any real statistic is computed.

## 9. Known limitations (stated in advance)
- Human predictors are norm-feature spaces, not human pairwise similarity judgements (SimLex-999 covers too few items).
- Frames contain the target word; contextual-meaning (B) vs computational-function (C) explanations are not separated here.
- Non-emotion adjectives include interoceptive body states (hungry, thirsty, drunk), which are conceptually close to affect; reported as a sub-analysis (body-state vs external-property adjectives), not a gate.
