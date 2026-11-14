# Run 5 pre-registration (written and committed before the job was launched, 14 Nov 2026)

## A. Stronger trigger ("CAST-v2")
Problem: the submitted CAST gate (v1) was fitted on plain prompts only. On the code-mixed wrapper test (run 4) it fired less
often on wrapped prompts, so wrapped harmful compliance stayed high (Igbo en_frame 79%).

Recipe (no other knobs, no sweep):
- Calibration set = the same 238 calibrate-split prompts (all 4 languages) PLUS their two wrapper variants
  (`en_frame`, `ask_local`) for hau/ibo/yor, built exactly as in run 4. No eval-split prompt is used for fitting.
- SafeTune `CASTTrainer.calibrate` refit on that set (condition vector, condition layer and behavior vectors all refit).
- alpha = 0.25 (same as v1). Gate threshold = the value that makes at most 20% of the augmented calibrate benign prompts
  fire, maximising harmful fire rate (same rule as v1's `fpr0.2`).
- Evaluated once on the held-out eval split: 957 plain prompts + 1,440 wrapped prompts (2,397 total).
  Generation shortcut: steered generation is only run on prompts where the v2 gate fires; for prompts where it does not
  fire the model is unchanged, so the run-1 / run-4 base outputs are reused.

Adoption rule (fixed now): adopt v2 as the submitted repair only if, pooled over all 2,397 held-out prompts,
1. harmful compliance(v2) <= harmful compliance(v1) - 3.0 percentage points, AND
2. over-refusal(v2) <= over-refusal(v1) + 2.0 percentage points.
Otherwise v1 stays the submitted repair and v2 is reported as a negative/partial result.

## B. CircuitKIT pass
CircuitKIT 0.2.2 (its TransformerLens port supports tiny-aya / cohere2). EAP-IG node-level discovery on a
refusal-vs-compliance contrast (SafetyPromptAdapter `harmful_vs_benign`: chat-templated harmful calibrate prompt vs a same-language benign calibrate prompt,
answer tokens = the language's most common first refusal token vs the compliance token `**`), 24 prompts per language.
Output: top-ranked heads/MLPs per language, overlap across languages and with the CAST layers (condition 10, behavior 16-20).
Descriptive only; it does not change the repair. Hard time limit 25 minutes; if it OOMs or fails on the T4 the attempt is
documented and skipped.

## Budget
Single t4-medium job, 70-minute cap (max $0.70). Previous spend about $1.28; project cap about $3.30.

## Addendum (after run 5, before run 6)
Run 5a result: CAST-v2 NOT adopted under the rule above (pooled HC 41.4% -> 33.0%, but OR 11.0% -> 19.0%, +8.0 pts > +2.0).
Run 5b: CircuitKIT EAP-IG ran out of T4 memory on the first language (activation-difference tensor for ~430-token
chat-templated prompts). Run 6 retries once with the same contrast but short Cohere chat turns without the default
system preamble (~50 tokens), 20-minute limit, job cap 25 minutes (max $0.25). If it fails again, CircuitKIT is skipped.
