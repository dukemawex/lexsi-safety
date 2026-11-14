#!/usr/bin/env python3
"""Run 5b: CircuitKIT EAP-IG node-level discovery of the refusal-vs-compliance circuit, per language.

Contrast (SafetyPromptAdapter, harmful_vs_benign, Arditi et al. style): clean = chat-templated harmful calibrate prompt with
answer = the language's most common first refusal token (from run-1 base outputs); corrupt = a benign calibrate prompt in the
same language with answer = the compliance token "**". (answer_contrastive is rejected by CircuitKIT: identical prompts.)
Runs in its own venv (circuitkit pins huggingface_hub<2 and transformer-lens 3.8). Descriptive only. See docs/PREREG_RUN5.md.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

REFUSAL_TOKEN = {"eng": "I'm", "hau": "Yi", "ibo": "Ọ", "yor": "Ma"}
COMPLY = "**"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="CohereLabs/tiny-aya-earth")
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/run5/circuitkit"))
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--langs", default="eng,yor,hau,ibo")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--ig-steps", type=int, default=3)
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--no-preamble", action="store_true",
                   help="Cohere turn tokens without the ~370-token default system preamble (the full template OOMs on a T4)")
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import circuitkit as ck
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task

    t0 = time.time()
    model = ck.load_model(a.model, dtype=a.dtype)
    tok = model.tokenizer
    print(f"loaded {a.model} in {time.time() - t0:.0f}s; layers={model.cfg.n_layers} heads={model.cfg.n_heads}", flush=True)
    rows = [json.loads(x) for x in a.splits.open(encoding="utf-8")]
    summary = {"model": a.model, "prompt_format": "no_preamble" if a.no_preamble else ("raw" if a.no_chat else "chat_template"), "algorithm": "eap-ig", "level": "node", "n_per_language": a.n, "languages": {}}
    for lang in a.langs.split(","):
        ref = REFUSAL_TOKEN.get(lang, " I") if not a.no_chat else " I"
        comply = COMPLY if not a.no_chat else " Sure"
        prompts = [r["prompt"] for r in rows if r["split"] == "calibrate" and r["language"] == lang and r["category"] != "benign"][: a.n]
        benign = [r["prompt"] for r in rows if r["split"] == "calibrate" and r["language"] == lang and r["category"] == "benign"]
        if a.no_preamble:
            def chat(qs):
                return ["<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + q + "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>" for q in qs]
            prompts, benign = chat(prompts), chat(benign)
        elif not a.no_chat:
            def chat(qs):
                qs = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
                bos = tok.bos_token or ""
                return [q[len(bos):] if bos and q.startswith(bos) else q for q in qs]  # TL prepends BOS itself
            prompts, benign = chat(prompts), chat(benign)
        ids = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in (("refusal", ref), ("comply", comply))}
        info = {"refusal_token": ref, "compliance_token": comply, "token_ids": ids, "n": len(prompts)}
        try:
            ds = SafetyPromptAdapter().adapt([{"prompt": q} for q in prompts], name=f"tuka-refusal-{lang}",
                                             refusal_token=ref, compliance_token=comply, pairing_mode="harmful_vs_benign",
                                             benign_controls=benign)
            register_task(NormalizedTaskSpec(ds, name=f"tuka-refusal-{lang}"))
            t = time.time()
            circ = ck.discover(model, f"tuka-refusal-{lang}", algorithm="eap-ig", level="node", n_examples=len(prompts),
                               batch_size=1, sparsity=0.9, scope="both", output_path=str(out / f"circuit_{lang}.pt"),
                               ig_steps=a.ig_steps)
            top = circ.top_nodes(40)
            info.update({"seconds": round(time.time() - t, 1), "top_nodes": {str(k): float(v) for k, v in dict(top).items()}})
            print(f"[{lang}] done in {info['seconds']}s; top10 {list(info['top_nodes'])[:10]}", flush=True)
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"[:2000]
            info["traceback"] = traceback.format_exc()[-4000:]
            print(f"[{lang}] FAILED: {info['error']}", flush=True)
            if "out of memory" in str(e).lower():
                summary["languages"][lang] = info
                break
        summary["languages"][lang] = info
        (out / "circuitkit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if torch.cuda.is_available():
        summary["max_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out / "circuitkit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "traceback"} for k, v in summary["languages"].items()}, ensure_ascii=False)[:6000], flush=True)


if __name__ == "__main__":
    main()
