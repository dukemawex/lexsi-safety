# Abstract

**Closing Tiny Aya Earth's Yoruba, Hausa and Igbo safety gap with conditional steering**

Small multilingual chat models are entering public-facing assistants in Nigeria, but safety training is mostly English. If refusals do not transfer, translation becomes the cheapest jailbreak. We measure this for Cohere Labs' Tiny Aya Earth (3.35B), the regional model recommended for African languages, and repair it without retraining.

We chain three Lexsi kits. CuratorKIT ingests TukaBench (harmful, culturally adapted harmful, and topic-matched benign prompts in English, Hausa, Igbo and Yoruba), cleans and deduplicates it with provenance, and we split it by source item so no translation of an eval prompt appears in calibration (957 eval, 238 calibration prompts). AuditKIT logs every stage as a per-language run and compares baseline with repair. SafeTune's CASTTrainer fits a conditional activation-steering repair on the calibration split, pooled across all four languages: a prompt-level gate at layer 10 and steering vectors at layers 16 to 20.

On held-out prompts the base model complies with harmful requests 18.9% of the time in English, 41.5% in Hausa, 49.4% in Igbo and 72.5% in Yoruba, rising to 90% for culturally adapted Yoruba prompts. It understands the prompts and, when it refuses, uses fluent native refusals: the gap is how often it refuses, not whether it can.

SafeTune's default gate threshold over-fires (50% of benign calibration prompts), giving 7.8% harmful compliance but 55.5% over-refusal on the eval split. We re-tuned the gate to a 20% benign false-fire budget and picked the steering strength on the calibration split with a fixed rule. The submitted repair lowers harmful compliance from 45.6% to 28.1% across the four languages (−17.6 points, 95% paired bootstrap CI [−20.7, −14.6], McNemar p < 1e−30), and in Yoruba from 72.5% to 40.6%. The Yoruba to English gap shrinks from 53.6 to 24.2 points. The new refusals are the model's own native refusal templates, and every prompt the gate leaves alone gets byte-identical output. The cost is higher over-refusal on topic-matched benign prompts (8.2% to 19.1%), which we trace to the gate firing on benign look-alikes. Belebele reading comprehension is unchanged in English, Hausa and Igbo and one question lower in Yoruba. Appending "Please answer in Igbo" to English requests raises compliance to 84%; the repair cuts this to 58% but barely helps English-framed Igbo prompts, an open gap.

Everything ran on one T4 through Hugging Face Jobs: about 2.1 GPU-hours, $1.28. At the measured 152 to 215 tokens per second, the self-hosted repaired model costs about $0.11 per 1,000 replies, similar to the cheapest closed APIs, while keeping the safety layer local and adjustable.

Limitations: refusal is scored by a multilingual lexicon without a native-speaker agreement study yet, natural code-switching and Nigerian Pidgin are not covered, and calibration is small. Next: native-speaker labels, a stronger gate calibrated on wrapped prompts, and CircuitKIT analysis.

Tracks: 2a problem 4 (multilingual and code-mixed safety), 2b problem 8 (inference-time safety steering).
