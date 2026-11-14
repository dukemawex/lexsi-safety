# Dataset and license notes

## Primary eval and contrast set

- [McGill-NLP/tukabench](https://huggingface.co/datasets/McGill-NLP/tukabench), CC-BY-NC 4.0, research only.
- Human-translated and culturally adapted JailbreakBench prompts in English, Hausa, Igbo, Yoruba (and four other African languages we can ignore).
- 100 harmful, 100 benign (over-refusal), 100 culturally adapted harmful prompts per language.
- Cite Akinode et al. 2026 and Chao et al. 2024 (JailbreakBench, MIT).

This is enough for Yoruba, Hausa, Igbo, English, and for a deterministic code-mixed set built from paired rows. It is not a license to republish the harmful text in our README or in commit history.

## Nigerian Pidgin

- Anyanwu et al. 2026, [arXiv:2610.39590](https://arxiv.org/abs/2610.39590), built 400 English prompts translated into Yoruba and Nigerian Pidgin (1,200 total). The paper says the authors plan to release the benchmark in future work. It is not a public, license-checked dataset we can use today.
- Do not write replacement harmful Pidgin prompts for this hackathon.
- If a public benign Pidgin corpus is added later, use it only for over-refusal or capability, and record its license first.

## Capability holdout (optional)

- Prefer a small, already public multilingual set that covers Hausa, Igbo and Yoruba, such as a FLORES-200 or Belebele slice. Confirm the license before publishing any derived files.

## What we will not do

- No new harmful content generation beyond loading standard red-team eval prompts from existing public benchmarks.
- No CuratorKIT adversarial generation task for this project.
- No closed-model distillation of harmful completions.
