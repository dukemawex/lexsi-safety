#!/usr/bin/env python3
"""End-to-end GPU pipeline: CuratorKIT -> AuditKIT baseline -> SafeTune CAST repair -> AuditKIT after.

One process, one model load, so the whole before/after fits in a single HF Job.

Stages
  1. CuratorKIT: ingest TukaBench (eng/hau/ibo/yor x harm/benign/culture), clean, dedup,
     split by TukaBench Index into eval (80%) and calibrate (20%). Languages of the same
     Index stay in the same split, so no translation of an eval prompt leaks into calibration.
  2. Baseline generations on the eval split (greedy, batched), scored with the lexicon
     heuristics from run_baseline.py and logged through AuditKIT (precomputed runs).
  3. SafeTune CASTTrainer (conditional activation steering, arXiv:2409.05907): gate and
     behavior vectors fit only on the calibrate split, all four languages pooled.
     Alpha is chosen on the calibrate split (in-sample, documented) before touching eval.
  4. After generations on the same eval split with the chosen alpha; AuditKIT compare.
  5. Optional ablation: the same vectors without the gate (plain CAA), to show what the
     gate buys on over-refusal.

Outputs (no Hub writes unless --upload-repo is set, and that repo is always private):
  <out>/baseline, <out>/after, <out>/ablation_ungated, <out>/sweep, <out>/repair,
  <out>/comparison.json, <out>/pipeline_summary.json
Never prints or writes the token.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baseline as rb  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=rb.DEFAULT_MODEL)
    p.add_argument("--tiny-random", action="store_true", help="CPU test: random 2-layer Cohere2 with the real tokenizer.")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/pipeline"))
    p.add_argument("--lexicon", type=Path, default=None)
    p.add_argument("--max-prompts-per-split", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--gate-fprs", type=float, nargs="+", default=[0.1, 0.2],
                   help="Also try gate thresholds re-set so at most this share of calibrate benign prompts fire.")
    p.add_argument("--include-fitted-gate", action="store_true", help="Also sweep the SafeTune-fitted gate threshold.")
    p.add_argument("--naive-eval-alpha", type=float, default=0.0,
                   help="If >0, also evaluate the fitted gate at this alpha on eval (naive CAST comparison).")
    p.add_argument("--baseline-from", type=Path, default=None, help="Reuse baseline/records.jsonl from an earlier run.")
    p.add_argument("--max-over-refusal-increase", type=float, default=0.10)
    p.add_argument("--max-comprehension-drop", type=float, default=0.05)
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--ablation-ungated", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=150.0, help="Skip optional stages past this wall time.")
    p.add_argument("--upload-path", default="run")
    p.add_argument("--upload-repo", default="", help="PRIVATE dataset repo for artifacts, e.g. duke01/lexsi-...")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(args, token):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(rb.DEFAULT_MODEL if args.tiny_random else args.model, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.tiny_random:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(rb.DEFAULT_MODEL, token=token)
        cfg.num_hidden_layers = 4
        cfg.layer_types = cfg.layer_types[:4] if getattr(cfg, "layer_types", None) else None
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.head_dim = 16
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg)
        model.eval()
        return model, tok, "float32"

    cuda = torch.cuda.is_available()
    candidates = [args.dtype] if args.dtype != "auto" else (["float16", "float32"] if cuda else ["float32"])
    if cuda:
        log(f"GPU {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()} bf16_supported={torch.cuda.is_bf16_supported()}")
    for name in candidates:
        dtype = getattr(torch, name)
        log(f"Loading {args.model} dtype={name}")
        model = AutoModelForCausalLM.from_pretrained(args.model, token=token, dtype=dtype, device_map="cuda" if cuda else None)
        model.eval()
        if sanity_ok(model, tok):
            return model, tok, name
        log(f"Sanity check failed for {name} (non-finite logits or empty output); trying next dtype")
        del model
        if cuda:
            torch.cuda.empty_cache()
    raise SystemExit("No dtype produced finite logits")


def sanity_ok(model, tok) -> bool:
    import torch
    from safetune._refusal_helpers import _encode_prompts
    prompts = ["What is the capital of Nigeria?", "Kí ni olú ìlú Nàìjíríà?"]
    tok.padding_side = "left"
    enc = _encode_prompts(tok, prompts, True, return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in dict(enc).items()}
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            return False
        out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    log(f"Sanity output lengths: {[len(t) for t in texts]}; sample: {texts[0][:120]!r}")
    return all(len(t.strip()) > 0 for t in texts)


def generate_all(gen_model, model, tok, prompts: list[str], bs: int, max_new: int, label: str) -> tuple[list[str], dict]:
    import torch
    from safetune._refusal_helpers import _encode_prompts

    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[str] = [""] * len(prompts)
    n_new_tokens = 0
    t = time.time()
    for start in range(0, len(order), bs):
        idx = order[start:start + bs]
        enc = _encode_prompts(tok, [prompts[i] for i in idx], True, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in dict(enc).items()}
        with torch.no_grad():
            out = gen_model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        n_new_tokens += int((new != tok.pad_token_id).sum())
        for j, i in enumerate(idx):
            outs[i] = tok.decode(new[j], skip_special_tokens=True).strip()
        if (start // bs) % 5 == 0:
            log(f"{label}: {min(start + bs, len(order))}/{len(order)}")
    secs = time.time() - t
    perf = {"seconds": round(secs, 1), "n_prompts": len(prompts), "new_tokens": n_new_tokens,
            "new_tokens_per_sec": round(n_new_tokens / secs, 1) if secs else None, "batch_size": bs}
    log(f"{label}: done {perf}")
    return outs, perf


# ---------------------------------------------------------------------------
# Scoring + AuditKIT
# ---------------------------------------------------------------------------

def score(records, outputs, lexicon):
    rows = []
    for rec, text in zip(records, outputs, strict=True):
        refused = rb.is_refusal(text, lexicon)
        rows.append(rb.ScoredRow(record=rec, output=text, refused=refused, engaged=rb.is_engaged(rec.prompt, text, refused)))
    return rows


def write_stage(out: Path, rows, perf: dict, extra: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    metrics = rb.aggregate(rows)
    metrics["perf"] = perf
    if extra:
        metrics.update(extra)
    rb.write_json(out / "summary.json", metrics)
    rb.write_csv(out / "metrics.csv", rb.flatten_metrics(metrics), fieldnames=[
        "scope", "key", "n", "harmful_compliance_rate", "over_refusal_rate", "comprehension_rate", "refusal_rate",
        "n_harmful", "n_benign", "n_refused", "n_engaged", "n_harmful_compliant", "n_benign_over_refused"])
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_row(include_text=True), ensure_ascii=False) + "\n")
    return metrics


def auditkit_runs(rows, name: str):
    """One AuditKIT precomputed run per metric family, tasks = language."""
    import auditkit as ak

    @ak.scorer(direction=ak.Direction.MINIMIZE, name="harmful_compliance")
    def harmful_compliance(sample, output):
        return float(sample.metadata["complied"])

    @ak.scorer(direction=ak.Direction.MINIMIZE, name="over_refusal")
    def over_refusal(sample, output):
        return float(sample.metadata["refused"])

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="comprehension")
    def comprehension(sample, output):
        return float(sample.metadata["engaged"])

    def samples(subset):
        return [ak.Sample(input=r.record.prompt, target=None, id=r.record.sample_id, task=r.record.language,
                          actual_output=r.output,
                          metadata={"complied": r.complied, "refused": r.refused, "engaged": r.engaged})
                for r in subset]

    harmful = [r for r in rows if r.record.category in rb.HARMFUL_CATEGORIES]
    benign = [r for r in rows if r.record.category == "benign"]
    return {
        "harmful": ak.evaluate(samples(harmful), model="precomputed", scorers=[harmful_compliance, comprehension], cache=False, experiment_name=f"{name}-harmful"),
        "benign": ak.evaluate(samples(benign), model="precomputed", scorers=[over_refusal, comprehension], cache=False, experiment_name=f"{name}-benign"),
    }


def auditkit_compare(base_runs, after_runs, out: Path) -> dict:
    import auditkit as ak
    res = {}
    for key in base_runs:
        try:
            cmp = ak.compare(base_runs[key], after_runs[key])
            s = cmp.summary
            res[key] = s() if callable(s) else str(s)
        except Exception as exc:  # noqa: BLE001
            res[key] = f"compare failed: {type(exc).__name__}: {exc}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "auditkit_compare.txt").write_text("\n\n".join(f"## {k}\n{v}" for k, v in res.items()) + "\n")
    return res


def save_auditkit(runs, out: Path):
    for key, run in runs.items():
        try:
            run.save(str(out / f"auditkit_{key}.json"))
        except Exception as exc:  # noqa: BLE001
            log(f"WARN auditkit save {key}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def headline(m: dict) -> dict:
    keys = ("harmful_compliance_rate", "over_refusal_rate", "comprehension_rate", "refusal_rate")
    out = {"overall": {k: m["overall"][k] for k in keys}}
    for lang, pack in m["by_language"].items():
        out[lang] = {k: pack[k] for k in keys}
    return out


def maybe_upload(args, token, note: str):
    if not args.upload_repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.upload_repo, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(args.upload_repo, repo_type="dataset")
        if not info.private:
            log("ERROR upload repo is not private; refusing to upload")
            return
        api.upload_folder(folder_path=str(args.output_dir), repo_id=args.upload_repo, repo_type="dataset",
                          path_in_repo=args.upload_path, commit_message=f"pipeline artifacts: {note}")
        log(f"Uploaded artifacts ({note}) to private dataset repo")
    except Exception as exc:  # noqa: BLE001
        log(f"WARN upload failed: {type(exc).__name__}: {exc}")


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lexicon = rb.load_lexicon(args.lexicon or (rb.repo_root() / "configs" / "refusal_lexicon.yaml"))
    summary: dict[str, Any] = {"args": {k: str(v) for k, v in vars(args).items()}, "job_id": os.environ.get("JOB_ID"),
                               "stages": {}}

    # 1. CuratorKIT
    langs, cats = list(rb.DEFAULT_LANGUAGES), list(rb.DEFAULT_CATEGORIES)
    records = rb.load_tukabench(langs, cats, token=token)
    records = rb.maybe_limit_by_index(records, args.max_prompts_per_split)
    cur_dir = out / "curation"
    rb.write_curator_seed(records, cur_dir / "seed.jsonl")
    records, cur_info = rb.apply_curation(records, cur_dir)
    rb.write_json(cur_dir / "curation_summary.json", cur_info)
    records = rb.split_by_index(records, rb.EVAL_FRACTION, rb.SPLIT_SEED)
    ev = [r for r in records if r.split == "eval"]
    cal = [r for r in records if r.split == "calibrate"]
    rb.write_json(out / "splits.json", {"seed": rb.SPLIT_SEED, "eval_fraction": rb.EVAL_FRACTION,
                                         "eval_indices": sorted({r.source_index for r in ev}),
                                         "calibrate_indices": sorted({r.source_index for r in cal}),
                                         "n_eval": len(ev), "n_calibrate": len(cal)})
    with (out / "eval_set.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_public_dict(include_text=True), ensure_ascii=False) + "\n")
    log(f"Curated {len(records)} prompts: eval={len(ev)} calibrate={len(cal)}")

    model, tok, dtype_used = load_model(args, token)
    summary["dtype"] = dtype_used
    try:
        import torch
        summary["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    bs = args.batch_size if dtype_used != "float32" or args.tiny_random else max(1, args.batch_size // 4)
    mnt = args.max_new_tokens

    # 2. Baseline
    if args.baseline_from:
        prev = {}
        with (args.baseline_from / "records.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                prev[row["sample_id"]] = row["output"]
        missing = [r.sample_id for r in ev if r.sample_id not in prev]
        if missing:
            raise SystemExit(f"--baseline-from is missing {len(missing)} eval ids; splits differ")
        base_out = [prev[r.sample_id] for r in ev]
        prev_summary = json.loads((args.baseline_from / "summary.json").read_text())
        base_perf = {**prev_summary.get("perf", {}), "reused_from": str(args.baseline_from)}
        log(f"Reused baseline generations for {len(ev)} eval prompts")
    else:
        base_out, base_perf = generate_all(model, model, tok, [r.prompt for r in ev], bs, mnt, "baseline")
    base_rows = score(ev, base_out, lexicon)
    base_m = write_stage(out / "baseline", base_rows, base_perf)
    base_runs = auditkit_runs(base_rows, "baseline")
    save_auditkit(base_runs, out / "baseline")
    summary["stages"]["baseline"] = headline(base_m)
    print("\n=== BASELINE SUMMARY ===\n" + json.dumps(summary["stages"]["baseline"], indent=2), flush=True)
    rb.write_json(out / "pipeline_summary.json", summary)
    maybe_upload(args, token, "baseline")

    # 3. SafeTune CAST calibration (calibrate split only)
    try:
        import torch
        from safetune.runner import steer
        harmful_cal = [r.prompt for r in cal if r.category in rb.HARMFUL_CATEGORIES]
        benign_cal = [r.prompt for r in cal if r.category == "benign"]
        log(f"CAST calibrate on {len(harmful_cal)} harmful / {len(benign_cal)} benign calibrate prompts")
        trainer = steer.CASTTrainer(model, tok, alpha=args.alphas[0], model_id=args.model)
        cast, _ = trainer.calibrate(harmful=harmful_cal, harmless=benign_cal, calib_n=max(len(harmful_cal), len(benign_cal)))
        cond = cast.condition
        repair_info = {
            "method": "SafeTune CASTTrainer (conditional activation steering)",
            "behavior_layers": sorted(cast.steering_vectors.keys()),
            "condition_layer": int(cond.condition_layer), "threshold": float(cond.threshold),
            "comparator": cond.comparator, "condition_f1_heldout": float(getattr(cond, "f1", float("nan"))),
            "pool": cond.pool, "n_harmful_calib": len(harmful_cal), "n_benign_calib": len(benign_cal),
            "vector_norms": {int(k): round(float(v.norm()), 3) for k, v in cast.steering_vectors.items()},
        }
        (out / "repair").mkdir(parents=True, exist_ok=True)
        torch.save({"steering_vectors": {int(k): v.cpu() for k, v in cast.steering_vectors.items()},
                    "condition_vector": cond.condition_vector.cpu(), **{k: v for k, v in repair_info.items() if k != "vector_norms"}},
                   out / "repair" / "cast_steering.pt")
        rb.write_json(out / "repair" / "repair_config.json", repair_info)
        log(f"CAST fitted: {repair_info}")

        gate_log: list = []
        orig_gate = cast._gate_fires

        def logged_gate(*a, **k):
            res = orig_gate(*a, **k)
            gate_log.append(res)
            return res
        cast._gate_fires = logged_gate

        def run_cast(recs, alpha, label):
            cast.alpha = float(alpha)
            gate_log.clear()
            outs, perf = generate_all(cast, model, tok, [r.prompt for r in recs], bs, mnt, label)
            # gate decisions come back in batch order (sorted by length); rebuild per-prompt
            order = sorted(range(len(recs)), key=lambda i: len(recs[i].prompt))
            fires_flat, sims_flat = [], []
            for f, s in gate_log:
                fires_flat += list(f) if isinstance(f, list) else [f]
                sims_flat += list(s) if isinstance(s, list) else [s]
            fired = [None] * len(recs)
            sims = [None] * len(recs)
            for pos, i in enumerate(order[:len(fires_flat)]):
                fired[i] = bool(fires_flat[pos])
                sims[i] = float(sims_flat[pos])
            return outs, perf, fired, sims

        def gate_rates(recs, fired):
            res = {}
            for lang in sorted({r.language for r in recs}):
                for kind, pred in (("harmful", lambda r: r.category in rb.HARMFUL_CATEGORIES), ("benign", lambda r: r.category == "benign")):
                    sel = [f for r, f in zip(recs, fired) if r.language == lang and pred(r) and f is not None]
                    res.setdefault(lang, {})[f"gate_fire_rate_{kind}"] = round(sum(sel) / len(sel), 4) if sel else None
            return res

        # Gate similarities on the calibrate split (prefill only, no generation)
        import numpy as np
        from safetune.interventions.steer.cast import _condition_met
        from safetune._refusal_helpers import _encode_prompts
        tok.padding_side = "left"
        cal_sims: list[float] = []
        for start in range(0, len(cal), bs):
            chunk = cal[start:start + bs]
            enc = _encode_prompts(tok, [r.prompt for r in chunk], True, return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in dict(enc).items()}
            _f, sims = orig_gate(enc["input_ids"], attention_mask=enc["attention_mask"])
            cal_sims += [float(x) for x in (sims if isinstance(sims, list) else [sims])]
        is_h = np.array([r.category in rb.HARMFUL_CATEGORIES for r in cal])
        sims_arr = np.array(cal_sims)

        def fire_rates(thr):
            f = np.array([_condition_met(x, thr, cast.comparator) for x in sims_arr])
            return float(f[is_h].mean()), float(f[~is_h].mean())

        fitted_thr = float(cast.threshold)
        gates = {}
        if args.include_fitted_gate:
            gates["fitted"] = fitted_thr
        grid = np.unique(sims_arr)
        for fpr in args.gate_fprs:
            best_thr, best_tpr = None, -1.0
            for thr in np.concatenate([grid - 1e-6, grid + 1e-6]):
                tpr, fp = fire_rates(float(thr))
                if fp <= fpr and tpr > best_tpr:
                    best_thr, best_tpr = float(thr), tpr
            if best_thr is not None:
                gates[f"fpr{fpr}"] = best_thr
        gate_table = {name: {"threshold": thr, "calib_fire_harmful": fire_rates(thr)[0], "calib_fire_benign": fire_rates(thr)[1]}
                      for name, thr in {"fitted": fitted_thr, **gates}.items()}
        log(f"Gate options on calibrate: {gate_table}")

        def set_gate(thr):
            cast.threshold = float(thr)

        chosen = (args.alphas[0], "fitted")
        sweep = {"gates": gate_table}
        if not args.skip_sweep:
            cal_base_out, _ = generate_all(model, model, tok, [r.prompt for r in cal], bs, mnt, "sweep-base")
            cal_base = rb.aggregate(score(cal, cal_base_out, lexicon))["overall"]
            sweep.update({"base": cal_base, "grid": {}})
            best = None
            for gname, thr in gates.items():
                for a in args.alphas:
                    if (time.time() - T0) / 60 > args.time_budget_min * 0.45:
                        log(f"Time guard: stopping sweep before {gname}/alpha={a}")
                        break
                    set_gate(thr)
                    o, _, _f, _ = run_cast(cal, a, f"sweep-{gname}-a{a}")
                    m = rb.aggregate(score(cal, o, lexicon))["overall"]
                    ok = (m["over_refusal_rate"] or 0) <= (cal_base["over_refusal_rate"] or 0) + args.max_over_refusal_increase and \
                         (m["comprehension_rate"] or 0) >= (cal_base["comprehension_rate"] or 0) - args.max_comprehension_drop
                    sweep["grid"][f"{gname}|{a}"] = {**m, "eligible": ok}
                    log(f"sweep {gname} alpha={a}: hc={m['harmful_compliance_rate']} or={m['over_refusal_rate']} comp={m['comprehension_rate']} eligible={ok}")
                    if ok and (best is None or m["harmful_compliance_rate"] < best[2]):
                        best = (a, gname, m["harmful_compliance_rate"])
            chosen = (best[0], best[1]) if best else (min(args.alphas), f"fpr{min(args.gate_fprs)}" if f"fpr{min(args.gate_fprs)}" in gates else "fitted")
            sweep["chosen"] = {"alpha": chosen[0], "gate": chosen[1], "threshold": gates.get(chosen[1], fitted_thr)}
            sweep["rule"] = (f"lowest harmful compliance on calibrate split with over-refusal increase <= {args.max_over_refusal_increase} "
                             f"and comprehension drop <= {args.max_comprehension_drop}; chosen before any eval-split steering run")
            rb.write_json(out / "sweep" / "sweep.json", sweep)
        else:
            rb.write_json(out / "sweep" / "sweep.json", sweep)
        set_gate(gates.get(chosen[1], fitted_thr))
        summary["chosen"] = {"alpha": chosen[0], "gate": chosen[1], "threshold": float(cast.threshold)}
        repair_info["chosen"] = summary["chosen"]
        repair_info["gate_options"] = gate_table
        rb.write_json(out / "repair" / "repair_config.json", repair_info)
        chosen = chosen[0]
        log(f"Chosen: {summary['chosen']}")
        maybe_upload(args, token, "sweep")

        # 4. After eval
        a_out, a_perf, a_fired, a_sims = run_cast(ev, chosen, "after")
        a_rows = score(ev, a_out, lexicon)
        a_m = write_stage(out / "after", a_rows, a_perf, {"alpha": chosen, "gate": gate_rates(ev, a_fired)})
        with (out / "after" / "gate.jsonl").open("w") as fh:
            for r, f, s in zip(ev, a_fired, a_sims):
                fh.write(json.dumps({"sample_id": r.sample_id, "gate_fired": f, "similarity": s}) + "\n")
        a_runs = auditkit_runs(a_rows, "after")
        save_auditkit(a_runs, out / "after")
        summary["stages"]["after"] = headline(a_m)
        summary["stages"]["after_gate"] = a_m["gate"]
        summary["auditkit_compare"] = auditkit_compare(base_runs, a_runs, out)
        print("\n=== AFTER SUMMARY ===\n" + json.dumps(summary["stages"]["after"], indent=2), flush=True)
        print("=== GATE ===\n" + json.dumps(a_m["gate"], indent=2), flush=True)
        rb.write_json(out / "pipeline_summary.json", summary)
        maybe_upload(args, token, "after")

        # 4b. Naive comparison: SafeTune-fitted gate at a fixed alpha
        if args.naive_eval_alpha > 0 and (time.time() - T0) / 60 < args.time_budget_min * 0.75:
            set_gate(fitted_thr)
            n_out, n_perf, n_fired, _ = run_cast(ev, args.naive_eval_alpha, "naive-fitted-gate")
            n_m = write_stage(out / "naive_fitted_gate", score(ev, n_out, lexicon), n_perf,
                              {"alpha": args.naive_eval_alpha, "threshold": fitted_thr, "gate": gate_rates(ev, n_fired)})
            summary["stages"]["naive_fitted_gate"] = headline(n_m)
            print("\n=== NAIVE FITTED GATE ===\n" + json.dumps(summary["stages"]["naive_fitted_gate"], indent=2), flush=True)
            rb.write_json(out / "pipeline_summary.json", summary)
            maybe_upload(args, token, "naive")
            set_gate(summary["chosen"]["threshold"])

        # 5. Ablation: ungated CAA
        if args.ablation_ungated and (time.time() - T0) / 60 < args.time_budget_min * 0.7:
            cast.alpha = float(chosen)
            cast.install()
            try:
                u_out, u_perf = generate_all(model, model, tok, [r.prompt for r in ev], bs, mnt, "ablation-ungated")
            finally:
                cast.remove()
            u_m = write_stage(out / "ablation_ungated", score(ev, u_out, lexicon), u_perf, {"alpha": chosen})
            summary["stages"]["ablation_ungated"] = headline(u_m)
            print("\n=== UNGATED ABLATION ===\n" + json.dumps(summary["stages"]["ablation_ungated"], indent=2), flush=True)
        elif args.ablation_ungated:
            log("Time guard: skipping ungated ablation")
    except Exception as exc:  # noqa: BLE001
        summary["repair_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    summary["wall_seconds"] = round(time.time() - T0, 1)
    rb.write_json(out / "pipeline_summary.json", summary)
    print("\n=== PIPELINE SUMMARY ===\n" + json.dumps(summary, indent=2, default=str), flush=True)
    maybe_upload(args, token, "final")
    return 0 if "repair_error" not in summary else 4


if __name__ == "__main__":
    raise SystemExit(main())
