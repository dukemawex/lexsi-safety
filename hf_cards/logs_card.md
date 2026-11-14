---
license: cc-by-nc-4.0
language: [en, ha, ig, yo]
tags: [safety, evaluation, auditkit, safetune, lexsi-hackathon]
---

# Tiny Aya Earth Nigerian-language safety: generations, metrics and traces

> **Content warning:** contains harmful requests and unedited model completions, some of them harmful (fraud, violence, weapons, self-harm, hate, misinformation), in English, Hausa, Igbo and Yoruba. Published as-is so every metric can be audited. Do not use to build harmful systems.

Outputs of the pipeline in https://github.com/dukemawex/lexsi-safety, run on Hugging Face Jobs (1x T4). Collections: [models](https://huggingface.co/collections/duke01/lexsi-tiny-aya-earth-nigerian-safety-models-6af879b97e17226835810729), [datasets](https://huggingface.co/collections/duke01/lexsi-tiny-aya-earth-nigerian-safety-datasets-6af879b902014765d7ed178d)

- `baseline/`: Tiny Aya Earth generations on the eval split, per-row refusal/engagement labels, metrics, AuditKIT runs.
- `after/`: same prompts with the submitted CAST repair, gate decisions and similarities (`gate.jsonl`).
- `naive_fitted_gate/`: SafeTune default gate threshold, alpha 0.5.
- `ablation_ungated/`: steering on every prompt, no gate.
- `retention/`: Belebele accuracy (200 questions per language) for base, CAST and ungated.
- `sweep/`, `repair/`: calibration sweep and repair config.
- `auditkit_compare.txt`: AuditKIT baseline to after comparison.
- `codemix/`: code-mixed wrapper variants (`en_frame`, `ask_local`), base and submitted CAST.
- `run5/`: pre-registered wrapper-aware gate (CAST-v2): `v2_plain/`, `v2_wrapped/` (records, gate traces), `repair_v2/`, `trigger_v2_summary.json` (not adopted), and `circuitkit/` (out-of-memory attempt).
- `run6/circuitkit/`: CircuitKIT retry (out of memory again).
- `figures/`: PNG and SVG figures.
- `job_logs/`: HF Jobs logs (tokens and secrets never printed).
- `labeling_sheet.csv`: 120 balanced rows for native-speaker judge validation (unlabelled).

Labels are from a multilingual refusal lexicon, not human annotation.
