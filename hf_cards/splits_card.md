---
license: cc-by-nc-4.0
language: [en, ha, ig, yo]
tags: [safety, tukabench, curatorkit, lexsi-hackathon]
---

# TukaBench Nigerian-language safety splits (CuratorKIT)

> Content warning: contains harmful requests (from TukaBench / JailbreakBench) in English, Hausa, Igbo and Yoruba.

Curated with Lexsi CuratorKIT (clean, exact dedup, schema gate, provenance) from `McGill-NLP/tukabench` configs `afri-jbb-harm`, `afri-jbb-culture`, `afri-jbb-benign`, languages `eng`, `hau`, `ibo`, `yor`. Split by TukaBench Index (seed 42): 80% eval (957 prompts), 20% calibrate (238). All translations of an item share a split.

Files: `eval_set.jsonl` (every prompt with split label, language, category, source index, prompt hash), `splits.json`, `curation/` (CuratorKIT manifest, provenance, checksums, dataset card).

Cite TukaBench (Akinode et al. 2026) and JailbreakBench (Chao et al. 2024). License CC-BY-NC-4.0 inherited from TukaBench.
