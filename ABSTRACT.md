# Abstract

**Closing Tiny Aya Earth's Yoruba, Hausa and Igbo safety gap with conditional steering**

Small multilingual chat models are entering Nigerian public-facing assistants, but safety training is mostly English, so translation may be the cheapest jailbreak. We measure this for Cohere Labs' Tiny Aya Earth (3.35B), the regional model recommended for African languages, and repair it without retraining.

We chain three Lexsi kits. CuratorKIT ingests TukaBench (harmful, culturally adapted harmful, and topic-matched benign prompts in English, Hausa, Igbo and Yoruba), cleans and deduplicates it with provenance, and we split it by source item so no translation of an eval prompt appears in calibration (957 eval, 238 calibration prompts). AuditKIT logs every stage as a per-language run and compares baseline with repair. SafeTune's CASTTrainer fits a conditional activation-steering repair on the calibration split, pooled across all four languages: a prompt-level gate at layer 10 and steering vectors at layers 16 to 20.

On held-out prompts the base model complies with harmful requests 18.9% of the time in English, 41.5% in Hausa, 49.4% in Igbo and 72.5% in Yoruba, rising to 90% for culturally adapted Yoruba prompts. It understands the prompts and refuses fluently when it does refuse: the gap is how often, not whether.

SafeTune's default gate over-fires (55.5% eval over-refusal), so we re-tuned the gate to a 20% benign false-fire budget and picked the steering strength on the calibration split with a fixed rule. The submitted repair lowers harmful compliance from 45.6% to 28.1% across the four languages (−17.6 points, 95% paired bootstrap CI [−20.7, −14.6], McNemar p < 1e−30), and in Yoruba from 72.5% to 40.6%. The Yoruba to English gap shrinks from 53.6 to 24.2 points. New refusals use the model's own native templates; prompts the gate leaves alone get byte-identical output. The cost is higher over-refusal on topic-matched benign prompts (8.2% to 19.1%), which we trace to the gate firing on benign look-alikes. Belebele reading comprehension is unchanged in English, Hausa and Igbo and one question lower in Yoruba. Appending "Please answer in Igbo" to English requests raises compliance to 84%; the repair cuts this to 58% but barely helps English-framed Igbo prompts, an open gap.

A pre-registered stronger trigger, refit on plain plus wrapped calibration prompts, cut pooled harmful compliance on 2,397 held-out plain and wrapped prompts from 41.4% to 33.0% but raised over-refusal from 11.0% to 19.0%; it failed our fixed adoption rule, so the submitted repair is unchanged. A CircuitKIT attribution pass ran out of T4 memory twice and was skipped.

Everything ran on one T4 through Hugging Face Jobs: about 2.6 GPU-hours, $1.59. The self-hosted repaired model costs about $0.11 per 1,000 replies, similar to the cheapest closed APIs, while keeping the safety layer local and adjustable.

Limitations: lexicon-scored refusal without a native-speaker agreement study yet, no natural code-switching or Nigerian Pidgin, small calibration set.

Tracks: 2a problem 4 (multilingual and code-mixed safety), 2b problem 8 (inference-time safety steering).
