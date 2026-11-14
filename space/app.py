"""Gradio demo: Tiny Aya Earth with and without the SafeTune CAST repair, side by side.

PREPARED, NOT DEPLOYED. Needs a GPU Space (T4 is enough), HF_TOKEN as a Space secret with access to the
gated CohereLabs/tiny-aya-earth, and the repair artifact (cast_steering.pt + repair_config.json) from the
model repo. Deploying it, and making it public, is Duke's decision.
"""
from __future__ import annotations

import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.steer.cast import CASTCondition, CASTModel
from safetune._refusal_helpers import _encode_prompts

MODEL_ID = "CohereLabs/tiny-aya-earth"
REPAIR_REPO = os.environ.get("REPAIR_REPO", "duke01/tiny-aya-earth-cast-nigerian-safety")
TOKEN = os.environ.get("HF_TOKEN")

tok = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
tok.padding_side = "left"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=TOKEN, dtype=dtype,
                                             device_map="cuda" if torch.cuda.is_available() else None).eval()
blob = torch.load(hf_hub_download(REPAIR_REPO, "cast_steering.pt", token=TOKEN), map_location="cpu")
cfg = json.load(open(hf_hub_download(REPAIR_REPO, "repair_config.json", token=TOKEN)))
cond = CASTCondition(condition_vector=blob["condition_vector"], condition_layer=int(blob["condition_layer"]),
                     threshold=float(cfg["chosen"]["threshold"]), comparator=blob["comparator"], pool=blob["pool"])
cast = CASTModel(model, {int(k): v for k, v in blob["steering_vectors"].items()}, condition=cond,
                 alpha=float(cfg["chosen"]["alpha"]), per_prompt_gate=True)


def run(prompt: str, max_new_tokens: int = 160):
    enc = _encode_prompts(tok, [prompt], True, return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in dict(enc).items()}
    kw = dict(max_new_tokens=int(max_new_tokens), do_sample=False, pad_token_id=tok.pad_token_id)
    with torch.no_grad():
        base = model.generate(**enc, **kw)
        fired, sim = cast._gate_fires(enc["input_ids"], attention_mask=enc["attention_mask"])
        rep = cast.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **kw)
    n = enc["input_ids"].shape[1]
    gate = f"Gate fired: {bool(fired[0])} (similarity {float(sim[0]):.3f}, threshold {cast.threshold:.3f})"
    return (tok.decode(base[0, n:], skip_special_tokens=True),
            tok.decode(rep[0, n:], skip_special_tokens=True), gate)


EXAMPLES = [
    ["Kí ni olú ìlú Nàìjíríà, kí o sì sọ ìtàn kúkúrú nípa rẹ̀?"],
    ["Write a short poem in Hausa about the harmattan."],
    ["Explain in Igbo how vaccines are tested for safety."],
]

with gr.Blocks(title="Tiny Aya Earth: Nigerian-language safety repair") as demo:
    gr.Markdown("## Tiny Aya Earth, before and after a conditional steering repair\n"
                "Content warning: this demo is for evaluating refusal behavior. The base model may comply with harmful "
                "requests, especially in Yoruba, Hausa and Igbo. Do not use outputs for anything harmful.\n\n"
                "Left: base model. Right: SafeTune CAST (the steering is added only when the prompt-level gate fires).")
    prompt = gr.Textbox(label="Prompt (English, Yoruba, Hausa or Igbo)", lines=3)
    mx = gr.Slider(32, 256, value=160, step=16, label="Max new tokens")
    btn = gr.Button("Generate")
    with gr.Row():
        out_b = gr.Textbox(label="Base Tiny Aya Earth", lines=10)
        out_r = gr.Textbox(label="With CAST repair", lines=10)
    gate = gr.Markdown()
    btn.click(run, [prompt, mx], [out_b, out_r, gate])
    gr.Examples(EXAMPLES, [prompt])

if __name__ == "__main__":
    demo.launch()
