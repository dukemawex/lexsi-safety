#!/usr/bin/env python3
"""Run 3: capability retention (Belebele MCQ) + ungated-CAA ablation, from the saved CAST artifact.

Loads repair/cast_steering.pt + repair/repair_config.json (the chosen gate threshold and alpha)
and rebuilds the SafeTune CASTModel without re-fitting, so this also checks the artifact loads.

Conditions
  base       : Tiny Aya Earth, no steering
  cast       : SafeTune CAST, tuned gate, chosen alpha (the submitted repair)
  ungated    : same behavior vectors and alpha applied to every prompt (plain CAA), no gate

Belebele (facebook/belebele, CC-BY-SA-4.0): first N parallel questions per language, prompt
with passage, question and 4 options, greedy short answer, parse the first digit 1-4.
Ablation: 'ungated' on the TukaBench eval split, to show what the gate buys on over-refusal.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baseline as rb  # noqa: E402
import run_pipeline as rp  # noqa: E402

BELEBELE = {"eng": "eng_Latn", "hau": "hau_Latn", "ibo": "ibo_Latn", "yor": "yor_Latn"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=rb.DEFAULT_MODEL)
    p.add_argument("--tiny-random", action="store_true")
    p.add_argument("--repair-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/retention"))
    p.add_argument("--belebele-n", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--max-prompts-per-split", type=int, default=0)
    p.add_argument("--lexicon", type=Path, default=None)
    p.add_argument("--upload-repo", default="")
    p.add_argument("--upload-path", default="run3")
    return p.parse_args()


def build_cast(model, repair_dir: Path):
    import torch
    from safetune.interventions.steer.cast import CASTModel, CASTCondition
    blob = torch.load(repair_dir / "cast_steering.pt", map_location="cpu")
    cfg = json.loads((repair_dir / "repair_config.json").read_text())
    chosen = cfg["chosen"]
    cond = CASTCondition(condition_vector=blob["condition_vector"], condition_layer=int(blob["condition_layer"]),
                         threshold=float(chosen["threshold"]), comparator=blob["comparator"], pool=blob["pool"])
    cast = CASTModel(model, {int(k): v for k, v in blob["steering_vectors"].items()}, condition=cond,
                     alpha=float(chosen["alpha"]), per_prompt_gate=True)
    return cast, chosen


def belebele_prompt(row) -> str:
    opts = "\n".join(f"{i}. {row[f'mc_answer{i}']}" for i in range(1, 5))
    return (f"{row['flores_passage']}\n\nQuestion: {row['question']}\n{opts}\n\n"
            "Answer with the number of the correct option only (1, 2, 3 or 4).")


def parse_choice(text: str):
    m = re.search(r"[1-4]", text or "")
    return int(m.group(0)) if m else None


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rp_args = argparse.Namespace(model=args.model, tiny_random=args.tiny_random, dtype=args.dtype)
    model, tok, dtype_used = rp.load_model(rp_args, token)
    cast, chosen = build_cast(model, args.repair_dir)
    rp.log(f"Rebuilt CAST from artifact: {chosen}")
    gate_log = []
    orig = cast._gate_fires

    def logged(*a, **k):
        k.setdefault("logits_to_keep", 1)  # gate only needs hidden states; avoid full-vocab logits (OOM on long prompts)
        r = orig(*a, **k)
        gate_log.append(r)
        return r
    cast._gate_fires = logged
    bs = args.batch_size
    summary = {"dtype": dtype_used, "chosen": chosen, "job_id": os.environ.get("JOB_ID"), "belebele": {}, "perf": {}}

    # Belebele retention
    from datasets import load_dataset
    items = []
    for lang, cfg in BELEBELE.items():
        ds = load_dataset("facebook/belebele", cfg, split="test", token=token)
        for row in list(ds)[: args.belebele_n]:
            items.append({"language": lang, "prompt": belebele_prompt(row), "gold": int(row["correct_answer_num"]),
                          "id": f"{cfg}-{row['question_number']}-{row['link'][-40:]}"})
    prompts = [it["prompt"] for it in items]
    results = {}
    for cond in ("base", "cast", "ungated"):
        gate_log.clear()
        if cond == "base":
            outs, perf = rp.generate_all(model, model, tok, prompts, bs, 6, f"belebele-{cond}")
        elif cond == "cast":
            outs, perf = rp.generate_all(cast, model, tok, prompts, bs, 6, f"belebele-{cond}")
        else:
            cast.install()
            try:
                outs, perf = rp.generate_all(model, model, tok, prompts, bs, 6, f"belebele-{cond}")
            finally:
                cast.remove()
        fired = []
        for f, _s in gate_log:
            fired += list(f) if isinstance(f, list) else [f]
        results[cond] = outs
        per = {}
        for lang in BELEBELE:
            idx = [i for i, it in enumerate(items) if it["language"] == lang]
            correct = sum(parse_choice(outs[i]) == items[i]["gold"] for i in idx)
            parsed = sum(parse_choice(outs[i]) is not None for i in idx)
            per[lang] = {"n": len(idx), "accuracy": round(correct / len(idx), 4), "k_correct": correct,
                         "parse_rate": round(parsed / len(idx), 4)}
        if cond == "cast" and fired:
            order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
            fired_by = [None] * len(prompts)
            for pos, i in enumerate(order[:len(fired)]):
                fired_by[i] = bool(fired[pos])
            for lang in BELEBELE:
                sel = [fired_by[i] for i, it in enumerate(items) if it["language"] == lang and fired_by[i] is not None]
                per[lang]["gate_fire_rate"] = round(sum(sel) / len(sel), 4) if sel else None
        summary["belebele"][cond] = per
        summary["perf"][f"belebele-{cond}"] = perf
        rp.log(f"belebele {cond}: {per}")
    with (out / "belebele_records.jsonl").open("w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            fh.write(json.dumps({"id": it["id"], "language": it["language"], "gold": it["gold"],
                                 **{f"out_{c}": results[c][i] for c in results}}, ensure_ascii=False) + "\n")
    rb.write_json(out / "retention_summary.json", summary)
    rp.maybe_upload(argparse.Namespace(upload_repo=args.upload_repo, upload_path=args.upload_path, output_dir=out), token, "belebele")

    # Ungated ablation on TukaBench eval split
    if not args.skip_ablation:
        lexicon = rb.load_lexicon(args.lexicon or (rb.repo_root() / "configs" / "refusal_lexicon.yaml"))
        records = rb.load_tukabench(list(rb.DEFAULT_LANGUAGES), list(rb.DEFAULT_CATEGORIES), token=token)
        records = rb.maybe_limit_by_index(records, args.max_prompts_per_split)
        records = rb.split_by_index(records, rb.EVAL_FRACTION, rb.SPLIT_SEED)
        ev = [r for r in records if r.split == "eval"]
        cast.install()
        try:
            u_out, u_perf = rp.generate_all(model, model, tok, [r.prompt for r in ev], bs, args.max_new_tokens, "ablation-ungated")
        finally:
            cast.remove()
        u_m = rp.write_stage(out / "ablation_ungated", rp.score(ev, u_out, lexicon), u_perf, {"alpha": chosen["alpha"]})
        summary["ablation_ungated"] = rp.headline(u_m)
        print("\n=== UNGATED ABLATION ===\n" + json.dumps(summary["ablation_ungated"], indent=2), flush=True)
    rb.write_json(out / "retention_summary.json", summary)
    print("\n=== RETENTION SUMMARY ===\n" + json.dumps(summary, indent=2, default=str), flush=True)
    rp.maybe_upload(argparse.Namespace(upload_repo=args.upload_repo, upload_path=args.upload_path, output_dir=out), token, "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
