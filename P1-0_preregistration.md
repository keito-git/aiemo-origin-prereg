# P1-0 Preregistration: Lexical-shadow pilot for H5 (Program "AI emotion structure differs from human emotion taxonomy", Project 1)

- Date: 2026-10-22 (committed before any P1-0 statistic is computed)
- Hypothesis card: Project 1 / H05 (lexical shadow, two layers)
- Purpose: go/no-go pilot. It decides whether the full P1 study of H5 proceeds. It is not a confirmatory test of "AI-specific emotion structure".

## 1. Data (all pre-existing; no new extraction)

- Source: AffectGeometry (Paper 1) per-model summaries `03_解析/per_model_v2/*.json`, 38 models, frozen revisions.
- Item set: 85 emotion words (`category_labels`), word-template condition (each word in 5 sentence templates; the stimulus contains the emotion word).
- Model RDM: `category_rdm` = cosine distance between category centroids at each model's representative layer, upper triangle (3,570 pairs). Stored separately for the `discovery` and `confirmation` stimulus splits.
- Lineages: registry v4 `lineage_id`. Main analysis uses lineages A–H (36 models). The two SUPPL models (QwQ-32B, Mistral-Small-3.2) are reported separately and excluded from all decision statistics.
- Human predictor H: Euclidean distance between words in NRC-VAD (V, A, D) from `code/stimuli/built/words_vad.csv`.
- Lexical predictor L: cosine distance between GloVe 6B 100d vectors (glove-wiki-gigaword-100). Multiword items ("worn out", "self-confident") use the mean of their token vectors (hyphenated items are split on "-" if the whole form is missing).

## 2. Measures

All correlations are Spearman. "Residualize" means OLS on ranks.

- M1 `r_H`: corr(model RDM, H) per model (discovery split).
- M2 `r_H|L`: partial correlation of model RDM and H controlling for L.
- M3 lexical share of human agreement: `1 - r_H|L / r_H` (median over main models).
- M4 residual `e_m`: rank(model RDM) residualized on rank(H) and rank(L).
- M5 residual sharing `S`: for each pair of distinct lineages, corr of the lineage-mean residuals (mean of `e_m` over models in the lineage); `S` = mean over the 28 lineage pairs. Discovery split.
- M6 null for M5: Mantel permutation (2,000 item permutations of one lineage's residual) per lineage pair; BH correction over the 28 pairs.
- M7 residual reliability `R`: per model, corr(e_m from discovery, e_m from confirmation); median over main models.
- M8 universality ratio `U = S / R`.
- M9 replication `C1`: corr(grand mean of lineage-mean residuals in discovery, same in confirmation).

## 3. Decision rule (fixed before analysis)

- **GO** (residual structure exists and is shared across lineages; P1 full study proceeds): `S >= 0.20` AND at least 80% of lineage pairs significant at BH q < 0.05 AND `U >= 0.5` AND `C1 >= 0.5`.
- **NO-GO** (residual is model-specific or noise in this data): `S < 0.10` OR `C1 < 0.3`. P1 then pivots (H5 layer (ii) not supported at the lexical-item level).
- Otherwise **INCONCLUSIVE**: report and redesign with richer stimuli (no-emotion-word situational texts).
- M1–M3 are descriptive (the "lexical shadow" part of H5); they are reported regardless and are not a gate.

## 4. What this pilot cannot show (stated in advance)

1. A shared residual is **not** evidence of AI-specific emotion structure. It may reflect contextual meaning (explanation B) or syntax, not a computational function (explanation C). P1-0 cannot separate B and C.
2. The stimuli contain the emotion word, so lexical information is maximal. This is the hardest setting for finding a non-lexical residual.
3. No non-emotion control domain is available in the saved data. The full study must add one (e.g., color and animal terms).
4. GloVe captures only static distributional statistics. A second lexical baseline may be added as a sensitivity analysis; it will not change the decision rule.
5. Representative layers differ across models (chosen in Paper 1).

## 5. Competing null for the whole program

Differences may be fully explained by emotion-word frequency and polysemy. P1-0 records GloVe vector norm (a frequency proxy) for each word and reports whether residual magnitude correlates with it (descriptive).
