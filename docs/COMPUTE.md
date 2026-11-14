# Compute plan

Today is Wednesday 11 November 2026. The submission deadline is Sunday 15 November 2026, 11:59 PM AoE, which is Monday 16 November 2026, 12:59 PM Africa/Lagos (WAT).

## Hardware we can use without spending

- Free Colab or Kaggle T4 16GB (sometimes a P100). The hackathon page says this can be enough for Tiny Aya.
- $20 in Hugging Face ZeroGPU or GPU-instance credits, only if Duke registered, joined Discord, and the Hugging Face username on the account matches the registration form. Credits are sent to the team lead in batches, first 800 teams, expire 30 days after redeem, and Hugging Face may ask for a payment method to redeem the coupon. Do not attach a card or spend beyond the coupon without explicit approval.
- Do not connect RunPod or any other paid GPU for this project unless Duke later approves it in so many words.

## Model footprint

- Tiny Aya family (Base, Global, Fire, Water, Earth): 3.35B parameters, about 6.7 GB of BF16 safetensors, 8K context, gated, CC-BY-NC-4.0 plus the Cohere Labs Acceptable Use Policy.
- Aya Expanse 8B: about 16 GB BF16, tight for training on a free T4. 32B is about 64 GB and is not feasible here. CircuitKIT also cannot yet run full-depth discovery on the 32B model.

## Estimated GPU hours (Tiny Aya Earth, T4-class)

- Download once: a few minutes, 6.7 GB.
- Baseline generation on about 1,200 TukaBench prompts (4 languages x 300) at 128 new tokens: roughly 1.5 to 3 hours, plus scoring. Using AuditKIT keyword and refusal detectors avoids loading a gated 7B judge (WildGuard is about 14.5 GB and is a poor default on 16GB).
- Steering calibration: minutes.
- After-repair eval: another 1.5 to 3 hours.
- Optional short SafeGrad LoRA: 2 to 4 hours.
- Optional CircuitKIT EAP-IG with a small example count: 1 to 4 hours if it fits; skip on OOM.

Total: about 6 to 12 GPU hours, which fits free-tier quotas plus the $20 credit if it arrives. Leave a reserve for a rerun after a bug.

## Colab gotchas from the kit docs

- SafeTune: `pip uninstall -y torchao` before importing SafeTune on Colab.
- Tiny Aya chat template adds a roughly 366-token system preamble. Leave `max_len` unset or set it to at least 512.
- Steering: if the automatic layer sweep is poor, set `pick_layer=24` (of 36).
- ReSta, if used: `alpha` around 0.25 on Tiny Aya; `alpha=1` breaks the model. Keep base and aligned copies on CPU if needed.

## Hugging Face Jobs pricing (checked 11 November 2026)

Jobs are the right product for this batch baseline. Spaces use the same hourly GPU prices but keep charging until the Space is paused. The cheapest relevant flavors are `t4-small` at $0.40 / hour and `t4-medium` at $0.60 / hour. Use `t4-medium` for the baseline (more system RAM). A 1 to 2 hour run is about $0.60 to $1.20, or $1.80 if it ran to a 3 hour timeout. See `docs/HF_JOBS.md` for the exact command. Do not launch until approved.
