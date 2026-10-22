# P1-0 Amendment 1 (2026-10-22, before any statistic is computed)

## Reason
A data check (label lists only; no statistic computed) showed two facts not known when the preregistration was written:
1. The Paper-1 `discovery` and `confirmation` splits are split **by word**, not by stimulus. Discovery = 85 words, confirmation = 86 different words (171 total, no overlap). Hence M7 (same-item residual reliability across splits) and M9 (same-pair replication) are undefined.
2. Some words have no NRC-VAD value in `words_vad.csv` (discovery: insulted, self-confident, self-conscious, worn out; confirmation: 11 words).

## Changes
- Items: words without a VAD value are excluded (discovery 81 words, confirmation 75 words).
- M7 replaced by **M7' within-lineage residual agreement**: mean corr of residuals between different models of the same lineage (lineages with >= 2 models), discovery set. It serves as the ceiling for between-lineage sharing.
- M8 becomes `U = S / M7'`.
- M9 replaced by **M9' independent-item replication**: the full M5/M6 analysis is repeated on the confirmation word set (independent items). `S_conf` and its fraction of significant lineage pairs are reported.
- Descriptive item-pair lists are reported for the discovery set only (they cannot be replicated on different words).

## Revised decision rule
- **GO**: `S_disc >= 0.20` AND >= 80% lineage pairs BH-significant (discovery) AND `U >= 0.5` AND `S_conf >= 0.20` AND >= 80% lineage pairs BH-significant (confirmation).
- **NO-GO**: `S_disc < 0.10` OR `S_conf < 0.10`.
- Otherwise **INCONCLUSIVE**.
All other parts of the preregistration are unchanged.
