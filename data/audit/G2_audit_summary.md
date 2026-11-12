# G2 support audit — summary (draft labels, to be verified by the team)

Rows: 200 pilot questions × top-3 BM25 windows = 600 windows (relation-stratified, dev-only).
Label rule (G2_AUDIT_README): 1 = window STATES subject + relation + a gold answer; 0 otherwise.
Labels drafted by Claude (AI); each has a written reason; unclear calls are flagged `uncertain`.
Report these as AI-drafted labels verified by the team, with the team's agreement statistic.

## Question level: share of questions with ≥1 supporting window (95% Wilson CI)

| measure | k / n | rate | 95% CI |
|---|---|---|---|
| top-1 | 40 / 200 | 20.0% | [15.0%, 26.1%] |
| top-2 | 54 / 200 | 27.0% | [21.3%, 33.5%] |
| top-3 (main) | 64 / 200 | 32.0% | [25.9%, 38.8%] |
| top-3, uncertain counted as 0 | 58 / 200 | 29.0% | [23.2%, 35.6%] |
| top-3, uncertain counted as 1 | 72 / 200 | 36.0% | [29.7%, 42.9%] |

## Window level

- Supporting windows: 100 / 600 (16.7%); flagged uncertain: 37.
- Automatic `answer_in AND subject_in` proxy vs labels: TP 81, FP 21, FN 19, TN 479 → precision 79.4%, recall 81.0%.
- `answer_in` alone: precision 71.9% (answer string present but fact not stated in 32 windows).
- Question-level agreement between proxy (any window a&s) and labels: 93.0%.

## By relation (top-3, main)

| relation | n | supported | rate |
|---|---|---|---|
| capital | 10 | 9 | 90% |
| occupation | 10 | 8 | 80% |
| sport | 9 | 6 | 67% |
| father | 10 | 6 | 60% |
| country | 15 | 6 | 40% |
| author | 23 | 8 | 35% |
| place of birth | 10 | 3 | 30% |
| genre | 25 | 6 | 24% |
| director | 25 | 5 | 20% |
| composer | 16 | 2 | 12% |
| screenwriter | 24 | 3 | 12% |
| producer | 21 | 2 | 10% |
| color | 2 | 0 | 0% |

## Observations

- Most misses are retrieval failures on short or generic titles (e.g. *Super*, *Stone*, *Money*, *Album*): the windows are about other entities, so the subject's own article is never retrieved.
- For several relations the relation word dominates the BM25 query: `screenwriter` questions repeatedly retrieve the same two filmography lists (Anatole de Grunwald, Crane Wilbur) regardless of the title.
- Same-name entities are a frequent false positive for the automatic proxy (Bread and Circuses: TV episode vs Star Trek episode; Harding County NM vs SD; Wiesen CH vs DE).
- The automatic answer-string match misses true support when the gold alias differs from the text (Máel Coluim III = Malcolm III; 'ski jumper' vs 'ski jumping'; surname-only 'Koenig').
- One question has an empty title ('Who was the screenwriter for ?', id in the CSV) — a PopQA data defect worth excluding.

## Suggested verification

Two teammates independently re-label a random 50 rows (150 windows) without looking at these labels; report Cohen's kappa against these labels and between each other. Resolve disagreements, then recompute the table above.
