#!/usr/bin/env python3
"""Stage the submission artifacts as PRIVATE Hugging Face repos under duke01. Never makes anything public.

Making them public and adding them to collections is a separate, manual step that needs Duke's approval.
"""
from pathlib import Path

from huggingface_hub import HfApi

R = Path("results/hfjob")
api = HfApi()
NS = "duke01"
plan = {
    ("model", f"{NS}/tiny-aya-earth-cast-nigerian-safety"): [
        ("hf_cards/model_card.md", "README.md"), (R / "run2/repair/cast_steering.pt", "cast_steering.pt"),
        (R / "run2/repair/repair_config.json", "repair_config.json"), (R / "run2/sweep/sweep.json", "sweep.json")],
    ("dataset", f"{NS}/tukabench-nigerian-safety-splits"): [
        ("hf_cards/splits_card.md", "README.md"), (R / "run2/eval_set.jsonl", "eval_set.jsonl"),
        (R / "run2/splits.json", "splits.json"), (R / "run2/curation", "curation")],
    ("dataset", f"{NS}/tiny-aya-earth-nigerian-safety-logs"): [
        ("hf_cards/logs_card.md", "README.md"), (R / "run/baseline", "baseline"), (R / "run2/after", "after"),
        (R / "run2/naive_fitted_gate", "naive_fitted_gate"), (R / "run2/sweep", "sweep"), (R / "run2/repair", "repair"),
        (R / "run2/auditkit_compare.txt", "auditkit_compare.txt"), (R / "run2/pipeline_summary.json", "pipeline_summary.json"),
        (R / "run3", "retention"), (R / "run4", "codemix"), (R / "job_logs", "job_logs"),
        (R / "labeling_sheet.csv", "labeling_sheet.csv"), ("results/RESULTS.md", "RESULTS.md"),
        ("results/metrics_table.csv", "metrics_table.csv"), ("results/paired_stats.json", "paired_stats.json")],
}
for (rtype, rid), items in plan.items():
    api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
    if not api.repo_info(rid, repo_type=rtype).private:
        raise SystemExit(f"{rid} is not private; refusing to upload")
    for src, dst in items:
        src = Path(src)
        if not src.exists():
            print(f"skip missing {src}")
            continue
        if src.is_dir():
            api.upload_folder(folder_path=str(src), path_in_repo=dst, repo_id=rid, repo_type=rtype)
        else:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=rid, repo_type=rtype)
    print(rid, "private:", api.repo_info(rid, repo_type=rtype).private, "files:", len(api.list_repo_files(rid, repo_type=rtype)))
