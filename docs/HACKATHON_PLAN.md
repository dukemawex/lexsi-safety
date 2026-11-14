# Recommended project plan

## Track and problem

Track 2a, problem 4: Multilingual and code-mixed safety. Test whether safety holds across languages and mixed-language prompts, then close the gaps you find.

Working title: measure and repair Tiny Aya Earth's safety gap on Nigerian languages (Yoruba, Hausa, Igbo, and English), including English-plus-local-language code-mixing, without inventing new harmful prompts.

## Why this one

It matches Duke's safety and mechanistic-interpretability background, uses a real public African-language safety set, and fits a 3.35B model on free GPUs in the days remaining. Tiny Aya Earth is the Cohere Labs regional chat model for West Asian and African languages, and Hausa, Igbo and Yoruba are in its official language list.

Two alternatives were considered and ranked lower for this week:

1. Track 2b problems 8 and 9 only (inference-time steering plus locating where safety lives), mostly in English. Cheaper, but less distinctive, and close in spirit to already-published African-language steering work.
2. Track 2a problem 2 as a dataset-and-report deliverable (for example a Pidgin or code-mixed eval set). Lighter compute, but the overall hackathon brief asks you to show that a small open-weights model becomes meaningfully better, and there is no public licensed Pidgin safety set to extend without writing new harmful prompts.

Track 1 problem 4 (low-resource multilingual fine-tuning with AlignTune) is feasible but is a capability project, not a safety one.

## Pipeline (three kits required, four preferred)

1. CuratorKIT: ingest TukaBench, clean, dedup, split, export trainer-ready and eval formats with provenance. No generation of new harmful items.
2. AuditKIT: baseline metrics on `hf:CohereLabs/tiny-aya-earth`.
3. SafeTune: `RefusalDirectionTrainer` as the default repair. Optional short `SafeGradTrainer` if steering is not enough.
4. CircuitKIT (optional but preferred): locate refusal-relevant structure with a small EAP or EAP-IG run on the experimental Tiny Aya port.
5. AuditKIT again: same held-out metrics, then `auditkit.compare`.

The handoff APIs exist in the official docs: CuratorKIT output folders load in AuditKIT (`load_hf`) and in SafeTune harden (`load_dataset` plus `SafeGradTrainer`). CircuitKIT `load_model` accepts Tiny Aya Hub ids and local SafeTune checkpoints.

## Metrics (more than one, before and after, per language)

1. Harmful-compliance rate on TukaBench harm and culture prompts.
2. Over-refusal rate on TukaBench benign prompts.
3. Comprehension or engagement rate, so a model that fails to understand Yoruba is not scored as safe.
4. A small public task-capability holdout, so steering has not destroyed usefulness.

Optional: a closed-vs-open ROI sketch, only with real prices and measured throughput.

## Day-by-day (Africa/Lagos, WAT)

- Wednesday 11 November: confirm registration, Hugging Face username, Discord, and Tiny Aya license acceptance. Freeze the plan. Keep this local scaffold private. Curate the eval split once Hub access is available.
- Thursday 12 November: baseline AuditKIT eval on Colab or Kaggle. Record GPU hours and tokens/sec.
- Friday 13 November: SafeTune steering. Start the after eval. If VRAM allows, a small CircuitKIT run. If steering fails, start a short SafeGrad LoRA.
- Saturday 14 November: finish after metrics, ablations (English vs each language vs code-mix), and the write-up draft.
- Sunday 15 November into Monday 16 November 12:59 PM WAT: public GitHub repo with a reproducible README, public Hugging Face collections for datasets, logs, and any adapters, write-up, and the submission form. Leave several hours of slack. Only publish when Duke says so.

## Key risks

- Registration closed on 8 November 2026, extended by 12 hours. If Duke did not register, organizers may reject a submission. Ask them on Discord or at hello@lexsi.ai immediately. Do not assume a late exception.
- Hugging Face credits require the username on the form to match the account that redeems them.
- Tiny Aya, and Aya Expanse, are gated. Accept the license before any GPU run.
- CircuitKIT on Tiny Aya is experimental and may OOM or fail the parity gate on a 16GB card.
- LLM-as-judge is unreliable in these languages. Do not make WildGuard or a closed judge the only scorer.
- Apparent safety in Yoruba can be a comprehension failure. That is already documented in Anyanwu et al. 2026. Report engagement alongside refusal.
- Nigerian Pidgin is a real gap: not listed in Tiny Aya's language card, and no public licensed safety set is available today.
- Several Lexsi kits are source-available under LSAL, not OSI open source. Academic and personal research use is allowed; do not productize them.
- The project must be developed during the hackathon window, which has already started. Keep the work clearly dated from this scaffold onward.
