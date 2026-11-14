---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-earth
language: [en, ha, ig, yo]
tags: [safety, activation-steering, cast, safetune, lexsi-hackathon, multilingual]
---

# Tiny Aya Earth + CAST safety repair for Yoruba, Hausa and Igbo

> **Content warning:** this artifact was built and evaluated on harmful requests (TukaBench). The companion logs dataset publishes full prompts and unedited model completions, including harmful ones.

A conditional activation-steering (CAST) adapter for `CohereLabs/tiny-aya-earth`, fit with Lexsi SafeTune `CASTTrainer` on 238 TukaBench calibration prompts pooled across English, Hausa, Igbo and Yoruba. It does not change model weights. At generation time a gate reads the prompt's hidden state at layer 10; if it fires, steering vectors are added at layers 16 to 20 with strength alpha = 0.25.

On 957 held-out TukaBench prompts, harmful compliance (lexicon-scored) goes from 45.6% to 28.1% across four languages, and from 72.5% to 40.6% in Yoruba. Over-refusal on topic-matched benign prompts rises from 8.2% to 19.1%. Full write-up: https://github.com/dukemawex/lexsi-safety. Collections: [models](https://huggingface.co/collections/duke01/lexsi-tiny-aya-earth-nigerian-safety-models-6af879b97e17226835810729), [datasets](https://huggingface.co/collections/duke01/lexsi-tiny-aya-earth-nigerian-safety-datasets-6af879b902014765d7ed178d)

## Files

- `cast_steering.pt`: `steering_vectors` {layer: tensor[2048]}, `condition_vector` tensor[2048], `condition_layer`, `comparator`, `pool`, fitted `threshold`.
- `repair_config.json`: chosen gate threshold and alpha (use these), gate options, layer norms.
- `sweep.json`: calibration-split sweep used to choose the configuration.
- `cast_v2_not_adopted/`: the pre-registered wrapper-aware gate (CAST-v2, refit on plain + wrapped calibration prompts). On 2,397 held-out plain and wrapped prompts it lowered pooled harmful compliance from 41.4% to 33.0% but raised over-refusal from 11.0% to 19.0%, failing the fixed adoption rule (over-refusal at most +2 points). Kept for reproducibility; **not the recommended repair.**

## Use

```python
import json, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.steer.cast import CASTCondition, CASTModel
repo = "duke01/tiny-aya-earth-cast-nigerian-safety"
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-earth"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-earth", dtype=torch.float16, device_map="cuda")
blob = torch.load(hf_hub_download(repo, "cast_steering.pt"), map_location="cpu")
cfg = json.load(open(hf_hub_download(repo, "repair_config.json")))["chosen"]
cond = CASTCondition(condition_vector=blob["condition_vector"], condition_layer=blob["condition_layer"],
                     threshold=cfg["threshold"], comparator=blob["comparator"], pool=blob["pool"])
cast = CASTModel(model, blob["steering_vectors"], condition=cond, alpha=cfg["alpha"], per_prompt_gate=True)
# cast.generate(input_ids=..., attention_mask=..., max_new_tokens=128, do_sample=False)
```

Encode prompts with the chat template as text and `add_special_tokens=False` (SafeTune `_encode_prompts`), as the gate was fit that way.

## Limitations

Heuristic lexicon judge with no native-speaker agreement study yet; over-refusal increase; code-mixing only via two fixed English wrappers, where the repair helps less (English-framed Igbo 80.6% -> 79.4%); no Pidgin coverage; one seed, greedy decoding. Measured behaviour on the eval set only. License follows the base model (CC-BY-NC-4.0 and the Cohere Labs AUP); built with SafeTune (LSAL, non-commercial research use).
