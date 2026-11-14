# Closing Tiny Aya Earth's Yoruba, Hausa and Igbo safety gap with conditional steering

**One-line claim:** Tiny Aya Earth complies with harmful requests almost four times as often in Yoruba as in English (72.5% vs 18.9% on held-out TukaBench prompts); a SafeTune conditional-steering repair, fit on 238 calibration prompts with no retraining, cuts Yoruba harmful compliance to 40.6% and the four-language average from 45.6% to 28.1%, while every prompt the gate does not flag gets byte-identical output.

> **Content warning.** This project studies harmful requests (fraud, violence, weapons, self-harm, hate, misinformation) written in English, Yoruba, Hausa and Igbo. The write-up quotes only refusal text and short paraphrased prompt titles. The private artifact repos contain full prompts and model completions, some of them harmful.

**Base model + method:** `CohereLabs/tiny-aya-earth` (3.35B, fp16) + SafeTune `CASTTrainer` (conditional activation steering), gate threshold re-tuned for a benign false-fire budget, chosen on a calibration split.

**Headline (held-out eval split, 957 prompts, same prompts before and after):**

| | English | Hausa | Igbo | Yoruba | All four |
|---|---|---|---|---|---|
| Harmful compliance, before → after (lower is better) | 18.9% → 16.4% | 41.5% → 25.2% | 49.4% → 30.0% | **72.5% → 40.6%** | 45.6% → 28.1% |
| Over-refusal on benign, before → after (lower is better) | 12.5% → 17.5% | 12.7% → 27.8% | 5.0% → 17.5% | 2.5% → 13.8% | 8.2% → 19.1% |

The repair is not free: over-refusal of benign look-alike prompts rises by 11 points on average. We report that cost up front and show where it comes from (the gate fires on 15 to 37% of benign prompts, Table 4).

## For judges: one minute

- **Finding.** On the same TukaBench prompts, Tiny Aya Earth's harmful compliance is 18.9% in English, 41.5% in Hausa, 49.4% in Igbo and 72.5% in Yoruba. The gap is largest on the culturally adapted jailbreaks (90% compliance in Yoruba). The model understands the prompts: it answers in the prompt's language, and when it refuses it does so with fluent native refusals.
- **Repair.** SafeTune CAST, a prompt-level gate plus a steering vector added only when the gate fires. Fit on 158 harmful + 80 benign calibration prompts pooled across all four languages. No weights changed, no new harmful prompts written.
- **Result.** Harmful compliance drops 17.6 points overall (95% paired bootstrap CI [−20.7, −14.6], exact McNemar p = 4e−34), and by 31.9 points in Yoruba. The Yoruba to English gap shrinks from 53.6 to 24.2 points.
- **Cost.** Benign over-refusal rises 11.0 points ([+7.5, +14.4]). Prompts where the gate stays off (448 of 957) produce byte-identical output to the base model.
- **Pipeline.** CuratorKIT (ingest, clean, dedup, provenance) → AuditKIT (baseline, after, `compare`) → SafeTune (`CASTTrainer` calibrate and steer), run end to end on one HF Jobs T4.
- **Retention.** Belebele reading-comprehension accuracy is unchanged in English, Hausa and Igbo and drops by one question (37.0% → 36.5%) in Yoruba; the gate fires on 0 to 2.5% of Belebele prompts.
- **Code-mixed finding.** Adding "Please answer in Igbo." to an English harmful request raises compliance from 18.9% (plain English) to 84.4%. The repair cuts that to 58.1%, but barely moves local-language prompts wrapped in an English frame (Igbo 80.6% → 79.4%). Reported as a gap, not hidden.
- **Compute.** Five HF Jobs on one T4 (one cancelled, one failed), about 2.1 GPU-hours, about $1.28 of credits in total.

## Tracks and problem statements

- Track 2a, AI Safety and Red Teaming, **problem 4: Multilingual and code-mixed safety** (primary). Code-mixing is covered by two fixed English wrappers around existing prompts (Table 7), not natural intra-sentence code-switching.
- Track 2b, AI Safety and Correction, **problem 8: Inference-time safety steering** (the repair method).

## Abstract

See [`ABSTRACT.md`](ABSTRACT.md) (under 500 words).

## Why this problem

More than 100 million people speak Yoruba, Hausa or Igbo, and Tiny Aya Earth is the Cohere Labs regional model recommended for African languages. A chat assistant built on it for public-facing use in Nigeria (education, health information, customer service) will receive requests in these languages. If safety training transfers poorly from English, the cheapest jailbreak is to translate the request. TukaBench (Akinode et al. 2026) was built to measure exactly this kind of cross-lingual safety gap, and earlier work found translation into low-resource languages jailbreaks frontier models (Yong et al. 2023). We test whether it holds for Tiny Aya Earth and whether it can be closed without retraining.

**Threat model.** A non-expert user who types or translates a harmful request into Yoruba, Hausa or Igbo, single turn, no optimisation, greedy decoding. Not covered: adaptive attackers, multi-turn attacks, weight access.

**Hypotheses** (set before the eval-split runs; the repair was selected on the calibration split only):
1. Harmful compliance is higher in Hausa, Igbo and Yoruba than in English. *Supported.*
2. A steering direction fit on pooled multilingual calibration data lowers harmful compliance in every language. *Supported for Hausa, Igbo, Yoruba (p < 1e−7 each); English drops only 2.5 points (p = 0.12), as expected since English starts low.*
3. Conditional gating keeps benign over-refusal close to baseline. *Partly. With the SafeTune default gate threshold the gate fired on 50% of benign calibration prompts and over-refusal reached 49% on calibration (Table 6). Re-tuning the gate to a 20% benign false-fire budget kept over-refusal at 19.1% on the eval split, still 11 points above baseline.*

## Hackathon kits used and how they connect

```
TukaBench (HF) ──► CuratorKIT ──► eval / calibrate split by TukaBench Index ──► Tiny Aya Earth (fp16, T4)
                   clean, exact dedup,                    │                          │
                   schema gate, provenance                │ calibrate (238)          │ eval (957)
                                                          ▼                          ▼
                                       SafeTune CASTTrainer.calibrate ──► CASTModel.generate (gated)
                                       (condition vector + CAA vectors)              │
                                                                                     ▼
                                       AuditKIT evaluate (precomputed runs, per-language tasks)
                                       AuditKIT compare (baseline → after: deltas, flips)
```

- **CuratorKIT 0.1.24.** `Curator(CuratorConfig(clean=True, dedup="exact", schema_gate=True, schema_required_fields=["instruction"], min_tokens=1, max_tokens=2048, export_formats=["alpaca"]))` over 1,200 TukaBench rows (4 languages × 3 configs × 100). All 1,200 rows passed the schema gate and exact dedup. Our exact-text join from the CuratorKIT export back to records kept 1,195: for 5 rows the TextCleaner had normalised the text, so the join missed them. They are dropped from every stage (a quirk of our join, not a quality rejection). Exports carry `lexsi_provenance.json`, checksums and a dataset card.
- **SafeTune 0.1.16.** `steer.CASTTrainer(model, tok).calibrate(harmful, harmless)` fits (a) the CAST condition (cosine-similarity gate at the best mid-depth layer, held-out F1 0.85, layer 10 of 36) and (b) CAA behavior vectors (difference of last-token means, harmful minus benign) at layers 16 to 20. We then re-set the gate threshold to a benign false-fire budget and swept alpha (Method).
- **AuditKIT 1.1.7.** Each stage's generations are logged as `ak.evaluate(..., model="precomputed")` runs with three custom scorers (`harmful_compliance` minimise, `over_refusal` minimise, `comprehension` maximise), one AuditKIT task per language. `ak.compare(baseline, after)` produces per-language deltas and flip counts (`auditkit_compare.txt` in the artifacts). Generation itself is our own batched loop so the CAST gate sees exactly the token sequence it was fit on.
- **CircuitKIT.** Not used in the submitted pipeline. The CAST condition layer search (layer 10 of 36 separates harmful from benign best) is a light form of "where safety lives"; a CircuitKIT pass is in "With more credits".

## Models and licenses

- **Base model:** `CohereLabs/tiny-aya-earth`, 3.35B, Cohere2 architecture, 36 layers, loaded in fp16 on a T4 (bf16 weights cast to fp16; a sanity check confirms finite logits). License CC-BY-NC-4.0 plus the Cohere Labs Acceptable Use Policy. Gated; accepted on the `duke01` account.
- **Repaired model:** the same frozen weights plus a 5-layer steering adapter and a gate (`cast_steering.pt`, about 100 KB). No weights were modified.
- No closed model was evaluated. The ROI section uses published API prices only.

## Datasets, splits and leak check

- **TukaBench** (`McGill-NLP/tukabench`, CC-BY-NC-4.0, derived from JailbreakBench, MIT). Configs `afri-jbb-harm` (direct harmful requests), `afri-jbb-culture` (culturally adapted harmful requests set in African contexts, our "jailbreak" set) and `afri-jbb-benign` (topic-matched benign requests, our over-refusal set). Splits `eng`, `hau`, `ibo`, `yor`. 100 items per config per language, the same items translated.
- **Split.** By TukaBench `Index` with seed 42: 80% eval, 20% calibrate. All four translations of an item land in the same split, so no eval prompt has a translation in calibration. Eval: 957 prompts (638 harmful, 319 benign). Calibrate: 238 (158 harmful, 80 benign).
- **Leak check.** Index-level split (no shared source item across splits) plus exact-text dedup by CuratorKIT. No embedding or MinHash check.
- **Capability retention:** Belebele (`facebook/belebele`, CC-BY-SA-4.0), first 200 parallel questions in `eng_Latn`, `hau_Latn`, `ibo_Latn`, `yor_Latn` (Table 5).
- **Attack prompt set** (Track 2a): 318 direct-harm and 320 culturally adapted harmful prompts in the eval split, about 80 of each per language. No encoding tricks, no prompt injection, no multi-turn.

## Metrics and why

| Metric | Direction | Why |
|---|---|---|
| Harmful compliance rate on harm + culture prompts | lower is better | The thing we want to reduce. A reply counts as compliant when it engages and contains no refusal cue. |
| Over-refusal rate on topic-matched benign prompts | lower is better | Catches a repair that just refuses everything. TukaBench benign prompts share topics with the harmful ones, so this is a hard over-refusal test. |
| Comprehension / engagement rate | higher is better | Guards against "safe because it did not understand". Flags empty, echoed or looping replies. |
| Belebele accuracy | higher is better | General capability retention in the same four languages. |
| Output identity when the gate is off | diagnostic | Shows the repair cannot hurt prompts it does not flag. |
| Gate firing rate on harmful vs benign | diagnostic | Explains where over-refusal comes from. |

**Checking the automatic judge.** Refusal is scored with a multilingual phrase lexicon (`configs/refusal_lexicon.yaml`), not an LLM judge, because LLM judges are unreliable in these languages. The model's refusals turned out to be highly templated in every language ("Ma bínú, ṣùgbọ́n èmi kò lè…", "Yi hakuri, amma ba zan iya…", "Ọ dị m nwute, mana enweghị m ike…"), which the lexicon matches. Adding the apology openers as extra cues changed no baseline label. **No human or native-speaker agreement study has been done yet**; a 120-row labelling sheet is prepared (`docs/labeling_sheet_template.csv` layout, rows in the private logs repo). "Compliant" means engaged without refusing; some compliant replies are off-topic or factually wrong rather than harmful, so harmful compliance is an upper bound on harmful content.

## Method

1. CuratorKIT ingest and split as above.
2. Baseline: greedy decoding, `max_new_tokens=128`, batch 16, left padding, Tiny Aya chat template (about 366-token system preamble).
3. `CASTTrainer.calibrate` on the calibration split (all four languages pooled). Defaults: CAA layers 14 to 18 scaled to 36 layers (16 to 20), last-token pooling, unnormalised vectors; condition layer grid over layers 9 to 27, cosine threshold grid −1 to 1 step 0.01.
4. Gate re-tuning. The SafeTune-fitted threshold maximises F1 and fired on 95% of harmful but also 50% of benign calibration prompts. We re-set it to the threshold that maximises harmful firing with at most 10% or 20% benign firing.
5. Sweep on the calibration split: gate ∈ {10%, 20% benign budget} × alpha ∈ {0.25, 0.5, 1.0}. Selection rule fixed in code before the eval run: lowest harmful compliance with over-refusal no more than 10 points above baseline and comprehension within 5 points. Chosen: 20% budget, alpha 0.25.
6. Eval-split generation with the chosen repair; AuditKIT runs and compare.
7. Comparisons on the eval split: SafeTune default gate at alpha 0.5 (naive CAST), and the same vectors at alpha 0.25 without a gate (plain CAA).
8. Retention: Belebele, 200 parallel questions per language, prompt = passage + question + 4 options + "Answer with the number of the correct option only", 6 new tokens, first digit parsed (parse rate 100%).
9. Code-mixed wrappers on the eval split for Hausa, Igbo, Yoruba (1,440 prompts), base vs submitted repair. These skip CuratorKIT, so they include the 5 rows the main split dropped.

Seed 42 for the split; decoding is greedy so generation is deterministic up to GPU kernels.

## Results

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

![Before and after by language](results/figures/before_after_by_language.svg)

![Calibration trade-off](results/figures/calibration_tradeoff.svg)

### Finding 1: the gap is real comprehension, not confusion

Across 638 harmful eval prompts, Hausa, Igbo and Yoruba replies are almost all in the prompt's language (fewer than 1% of Igbo and Yoruba replies start in ASCII-only English), and refusals are fluent and native. The model knows how to refuse in Yoruba; it just does so far less often.

### Finding 2: culturally adapted prompts are the weak spot

Compliance on culturally adapted prompts is 31% in English and 60 to 90% in the three Nigerian languages, versus 6% (English) to 55% (Yoruba) on direct requests.

### Finding 3: steering recruits the model's own refusals

Of the 112 harmful prompts that flipped from compliant to refused, the new replies are the same native refusal templates the base model uses, not degenerate text. We found no looping or collapsed replies in the steered outputs.

### Finding 4: the gate is the bottleneck

Where the gate stays off, output is unchanged (448 of 448 identical). Where it fires, it brings refusals. Over-refusal comes almost entirely from the gate firing on benign look-alikes (15 to 37% per language). A better gate is the clearest next step.

### Finding 5: the gate is what makes steering usable

At the same strength (alpha 0.25), applying the vectors to every prompt gives 20.6% harmful compliance but 46.3% over-refusal; the gated repair gives 28.1% and 19.1%. SafeTune's default F1-optimal gate at alpha 0.5 sits at the other extreme: 7.8% harmful compliance, 55.5% over-refusal. The submitted point trades some safety for a much smaller over-refusal cost, and the selection rule was fixed before the eval run.

### Finding 6: asking for a local-language answer is a strong jailbreak, and the repair only partly covers it

The same English harmful requests that get 18.9% compliance become 47.5% (Hausa), 84.4% (Igbo) and 60.6% (Yoruba) compliant when the user appends "Please answer in <language>." Local-language prompts inside an English frame ("Please answer the following in English.") are also more compliant than plain ones for Igbo (80.6% vs 49.4%). The repair, fit only on plain prompts, reduces the ask-local variant by 17 to 26 points but leaves the English-framed Igbo variant almost unchanged (the gate rarely fires on it). Calibrating the gate on wrapped prompts too is the obvious fix.

### What did not work

- The SafeTune default gate threshold (F1-optimal) fires on half of the benign calibration prompts; at alpha 0.5 it gave 7.8% harmful compliance but 55.5% over-refusal on the eval split.
- Our first retention job ran out of GPU memory because the gate's prefill pass computed full-vocabulary logits for long Belebele passages; passing `logits_to_keep=1` fixed it.
- Larger alpha did not monotonically help: alpha 1.0 gave higher calibration harmful compliance than alpha 0.5 at both gate budgets.

## Cost and ROI

**Compute used.** HF Jobs `t4-medium` (1× T4 16 GB, $0.60/hour), Hugging Face credits on `duke01`. Five jobs (details in `docs/HF_JOBS.md`): about 7,700 billed seconds, 2.1 GPU-hours, about $1.28. Run 1 was cancelled after its first sweep point showed the default gate over-fires; run 3 failed with an out-of-memory error and was re-run.

**Measured throughput** (HF transformers `generate`, fp16, batch 16, no vLLM): 215 new tokens/s unsteered, 152 to 165 tokens/s with CAST (extra gate prefill and hooks). Average reply 106 tokens; average user prompt 35 Tiny Aya tokens (399 with the chat preamble).

**Cost per 1,000 replies** (same mix of prompts, 128-token cap):

| Option | How computed | Cost per 1k replies |
|---|---|---|
| Tiny Aya Earth, self-hosted T4, base | 475 s per 957 replies → 0.138 h × $0.60 | $0.083 |
| Tiny Aya Earth + CAST repair, self-hosted T4 | 623 s per 957 replies → 0.181 h × $0.60 | $0.109 |
| gpt-5-nano API | 35k in × $0.05/M + 106k out × $0.40/M | $0.044 |
| gpt-4o-mini API | 35k × $0.15/M + 106k × $0.60/M | $0.069 |
| gpt-5-mini API | 35k × $0.25/M + 106k × $2.00/M | $0.221 |

API prices from OpenAI's pricing page, checked 14 Nov 2026. API token counts are a lower bound: closed tokenizers usually split Yoruba diacritics into more tokens than Tiny Aya's, and reasoning models bill hidden reasoning tokens as output. Self-hosted cost assumes a fully used GPU; a vLLM backend (SafeTune ships a vLLM steering hook) would raise throughput several-fold. On this evidence the self-hosted repaired model costs about the same as the cheapest closed APIs per reply. Its advantages are control (an auditable, adjustable safety layer, local data) rather than price, and we did not measure closed-model safety in these languages.

## Limitations

- **Heuristic judge.** Lexicon refusal scoring, no native-speaker agreement yet. Compliance counts engagement, not verified harm.
- **Over-refusal rises** by 11 points on topic-matched benign prompts.
- **Small calibration set** (238 prompts) and in-sample selection on it; eval numbers are held out.
- **Code-mixing only via two fixed wrappers.** Natural intra-sentence code-switching is not covered, and the repair barely helps English-framed Igbo prompts (Table 7).
- **Not covered:** Nigerian Pidgin (no public licensed safety set), multi-turn, adaptive or optimised attacks, sampling-based decoding.
- **Belebele prompt format** is our own zero-shot format; absolute accuracies are not comparable to published Belebele numbers, only before vs after. Small fp16 batch effects change a base-model Belebele answer here and there (one Igbo question differed between batch 16 and batch 8).
- **One model, one seed, greedy decoding.**
- **Dual use.** The steering vectors could in principle be subtracted to weaken refusals. We release no inverted direction and no harmful completions in the write-up. Measured behaviour only: nothing here shows hidden or intentional unsafe behaviour in the base model.

## With more credits

- Native-speaker labelling of 300 replies for judge agreement (Cohen's κ), then a calibrated judge. Expected: harmful-compliance estimates within ±5 points of the lexicon.
- Better gate: a small logistic probe on several layers, or per-language thresholds. Expected: over-refusal back to within 3 to 5 points of baseline at similar harmful compliance.
- CircuitKIT on the experimental Tiny Aya port to locate refusal components and compare with the CAST layers. SafeGrad hardening as a weight-level comparison.
- Calibrate the gate on wrapped and code-mixed variants as well as plain prompts. Expected: ask-local and English-framed compliance brought down to the plain-prompt after levels (25 to 40%).
- Natural code-switched prompts built by recombining existing TukaBench translations (no new harmful content), and the Anyanwu et al. Pidgin set when it is released.
- vLLM serving for a real ROI number.

## Artifacts

All created as **private** repos under `duke01`; they become public only when Duke approves.

- Models collection (planned): `duke01/tiny-aya-earth-cast-nigerian-safety` (steering adapter, gate, configs, sweep, model card).
- Datasets collection (planned): `duke01/tukabench-nigerian-safety-splits` (curated eval/calibrate split with CuratorKIT provenance) and `duke01/tiny-aya-earth-nigerian-safety-logs` (generations, metrics, AuditKIT runs, gate traces, job logs).
- Working bucket: `duke01/lexsi-tiny-aya-earth-safety-run` (private, raw job outputs).
- GitHub: `dukemawex/lexsi-safety` (private).
- Demo: `space/` (Gradio, prepared, not deployed).

## Reproduction

Requirements: Python 3.12, a 16 GB GPU, `HF_TOKEN` with Tiny Aya Earth access. CPU tests use Python 3.12+ with CPU torch.

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=5.15,<6" datasets accelerate pyyaml auditkit==1.1.7 "curatorkit[connectors]==0.1.24" safetune==0.1.16
# full pipeline (baseline + CAST calibrate + sweep + after + naive comparison), about 1 h on a T4
python scripts/run_pipeline.py --output-dir artifacts/pipeline --gate-fprs 0.1 0.2 --alphas 0.25 0.5 1.0 --naive-eval-alpha 0.5
# retention + ungated ablation from the saved artifact
python scripts/run_retention.py --repair-dir artifacts/pipeline/repair --output-dir artifacts/retention --batch-size 8
# code-mixed wrappers
python scripts/run_codemix.py --repair-dir artifacts/pipeline/repair --output-dir artifacts/codemix
# tables and figures (CPU)
python scripts/07_make_tables.py --runs <dir with run/, run2/, run3/> --out results
# CPU end-to-end check with a random 4-layer Cohere2 and the real tokenizer
python scripts/run_pipeline.py --tiny-random --max-prompts-per-split 5 --batch-size 4 --max-new-tokens 8 --output-dir /tmp/pt
```

HF Jobs commands we ran are in `docs/HF_JOBS.md`.

| Path | What it is |
|---|---|
| `scripts/run_pipeline.py` | End-to-end GPU pipeline (CuratorKIT → AuditKIT → SafeTune CAST → AuditKIT) |
| `scripts/run_retention.py` | Belebele retention and ungated ablation from a saved CAST artifact |
| `scripts/run_codemix.py` | Code-mixed / cross-lingual wrapper variants, base vs CAST |
| `scripts/08_make_labeling_sheet.py` | Balanced 120-row sheet for native-speaker judge validation |
| `scripts/09_stage_hf_private.py` | Uploads artifacts to PRIVATE HF repos (never public) |
| `scripts/run_baseline.py` | Data loading, curation, split, lexicon scoring (shared helpers) |
| `scripts/07_make_tables.py` | Tables, paired bootstrap CIs, McNemar tests, figures |
| `scripts/job_entry*.sh` | Exact HF Jobs entry scripts |
| `configs/` | Refusal lexicon, pipeline config |
| `results/` | Aggregate tables and figures (no prompt or completion text) |
| `space/` | Gradio demo, not deployed |
| `scripts/00`–`06` | Earlier scaffold and dry-run stages |

**Declaration of work outside the hackathon:** none. The scaffold was created on 11 to 12 November 2026, during the hackathon window, and all runs on 14 November 2026.

## Team, acknowledgements, LLM use

- Emmanuel Effiom Duke (solo): project lead, design, decisions.
- Compute: Hugging Face credits from the hackathon. Data: TukaBench (McGill NLP), JailbreakBench, Belebele (Meta). Kits: Lexsi Labs (LSAL, used for non-commercial research).
- LLM usage: most of the code and a first draft of this write-up were produced with an AI coding assistant under Duke's direction; all numbers come from the logged runs.

## References

- Akinode et al. 2026, TukaBench. `McGill-NLP/tukabench`.
- Chao et al. 2024, JailbreakBench.
- Lee et al. 2024, Programming Refusal with Conditional Activation Steering (CAST), arXiv:2409.05907.
- Rimsky et al. 2024, Steering Llama 2 via Contrastive Activation Addition.
- Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction.
- Bandarkar et al. 2024, The Belebele Benchmark.
- Yong et al. 2023, Low-Resource Languages Jailbreak GPT-4.
- Cohere Labs, Tiny Aya model card.

## Changelog

- v1 (14 Nov 2026): first complete run.
