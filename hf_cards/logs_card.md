---
license: cc-by-nc-4.0
language: [en, ha, ig, yo]
tags: [safety, evaluation, auditkit, safetune, lexsi-hackathon]
---

# Tiny Aya Earth Nigerian-language safety: generations, metrics and traces

> Content warning: contains harmful requests and some harmful model completions in English, Hausa, Igbo and Yoruba.

Outputs of the pipeline in https://github.com/dukemawex/lexsi-safety, run on Hugging Face Jobs (1x T4).

- `baseline/`: Tiny Aya Earth generations on the eval split, per-row refusal/engagement labels, metrics, AuditKIT runs.
- `after/`: same prompts with the submitted CAST repair, gate decisions and similarities (`gate.jsonl`).
- `naive_fitted_gate/`: SafeTune default gate threshold, alpha 0.5.
- `ablation_ungated/`: steering on every prompt, no gate.
- `retention/`: Belebele accuracy (200 questions per language) for base, CAST and ungated.
- `sweep/`, `repair/`: calibration sweep and repair config.
- `auditkit_compare.txt`: AuditKIT baseline to after comparison.
- `job_logs/`: HF Jobs logs (tokens and secrets never printed).
- `labeling_sheet.csv`: 120 balanced rows for native-speaker judge validation (unlabelled).

Labels are from a multilingual refusal lexicon, not human annotation.
