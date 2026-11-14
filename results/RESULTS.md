### Table 1. Baseline harmful compliance and over-refusal by language (held-out eval split)

| Language | Harmful compliance [95% CI] | Gap vs English | Direct harm | Culturally adapted | Over-refusal on benign [95% CI] | n harmful / benign |
|---|---|---|---|---|---|---|
| English | 18.9% [14, 26] |  | 6.3% | 31.2% | 12.5% [7, 22] | 159 / 80 |
| Hausa | 41.5% [34, 49] | +22.6 pts | 22.8% | 60.0% | 12.7% [7, 22] | 159 / 79 |
| Igbo | 49.4% [42, 57] | +30.5 pts | 37.5% | 61.3% | 5.0% [2, 12] | 160 / 80 |
| Yoruba | 72.5% [65, 79] | +53.6 pts | 55.0% | 90.0% | 2.5% [1, 9] | 160 / 80 |

### Table 2. Same model before and after, same held-out prompts (harmful compliance / over-refusal, lower is better)

| Language | Baseline | CAST tuned gate (submitted) | CAST default gate, alpha 0.5 | Ungated CAA, alpha 0.25 |
|---|---|---|---|---|
| English | 18.9% / 12.5% | 16.4% / 17.5% | 7.5% / 45.0% | 6.9% / 48.8% |
| Hausa | 41.5% / 12.7% | 25.2% / 27.8% | 3.1% / 67.1% | 19.4% / 51.2% |
| Igbo | 49.4% / 5.0% | 30.0% / 17.5% | 6.2% / 61.3% | 23.1% / 53.8% |
| Yoruba | 72.5% / 2.5% | 40.6% / 13.8% | 14.4% / 48.8% | 33.1% / 31.2% |
| All four | 45.6% / 8.2% | 28.1% / 19.1% | 7.8% / 55.5% | 20.6% / 46.2% |

### Table 3. Paired change, baseline to submitted repair (same prompts; 95% paired bootstrap CI; exact McNemar p)

| Language | Harmful compliance change [95% CI] | McNemar p | Over-refusal change [95% CI] | McNemar p |
|---|---|---|---|---|
| English | -2.5 pts [-5.0, -0.6] | 0.12 | +5.0 pts [+1.2, +10.0] | 0.12 |
| Hausa | -16.4 pts [-22.0, -10.7] | 3e-08 | +15.2 pts [+7.6, +24.1] | 0.00049 |
| Igbo | -19.4 pts [-25.6, -13.8] | 9.3e-10 | +12.5 pts [+6.2, +20.0] | 0.002 |
| Yoruba | -31.9 pts [-39.4, -24.4] | 8.9e-16 | +11.2 pts [+5.0, +18.8] | 0.0039 |
| All four | -17.6 pts [-20.7, -14.6] | 3.9e-34 | +11.0 pts [+7.5, +14.4] | 5.8e-11 |

### Table 4. Where the gate fires on the eval split (submitted repair)

| Language | Fires on harmful | Fires on benign |
|---|---|---|
| English | 58.5% | 15.0% |
| Hausa | 78.0% | 36.7% |
| Igbo | 63.7% | 20.0% |
| Yoruba | 68.8% | 28.7% |

### Table 5. Capability retention: Belebele reading comprehension accuracy (200 parallel questions per language, higher is better)

| Language | Baseline | CAST tuned gate (submitted) | Ungated CAA | Gate fires on Belebele |
|---|---|---|---|---|
| English | 83.0% | 83.0% | 82.5% | 0.0% |
| Hausa | 59.5% | 59.5% | 60.0% | 0.5% |
| Igbo | 44.0% | 44.0% | 44.5% | 1.0% |
| Yoruba | 37.0% | 36.5% | 36.0% | 2.5% |

### Table 7. Code-mixed and cross-lingual wrappers (eval split, harmful compliance / over-refusal, base → submitted CAST)

`en_frame`: local-language prompt inside an English instruction ("Please answer the following in English."). `ask_local`: the English source prompt plus "Please answer in <language>." Plain prompts from Table 2 for reference.

| Language | Plain prompt | en_frame | ask_local |
|---|---|---|---|
| Hausa | 41.5% / 12.7% → 25.2% / 27.8% | 48.1% / 8.8% → 37.5% / 10.0% | 47.5% / 1.2% → 30.0% / 5.0% |
| Igbo | 49.4% / 5.0% → 30.0% / 17.5% | 80.6% / 1.2% → 79.4% / 1.2% | 84.4% / 1.2% → 58.1% / 2.5% |
| Yoruba | 72.5% / 2.5% → 40.6% / 13.8% | 67.5% / 7.5% → 55.0% / 11.2% | 60.6% / 1.2% → 41.2% / 3.8% |

### Table 6. Calibration-split sweep used to choose the repair (in-sample; chosen before any eval-split steering)

| Gate | alpha | Harmful compliance | Over-refusal | Eligible |
|---|---|---|---|---|
| none (baseline) | 0 | 48.1% | 11.2% | |
| fpr0.1 | 0.25 | 34.8% | 15.0% | yes |
| fpr0.1 | 0.5 | 29.1% | 18.8% | yes |
| fpr0.1 | 1.0 | 37.3% | 18.8% | yes |
| fpr0.2 | 0.25 | 27.2% | 17.5% | yes |
| fpr0.2 | 0.5 | 17.1% | 27.5% | no |
| fpr0.2 | 1.0 | 27.2% | 27.5% | no |

Chosen: gate fpr0.2, alpha 0.25. Rule: lowest harmful compliance on calibrate split with over-refusal increase <= 0.1 and comprehension drop <= 0.05; chosen before any eval-split steering run.

