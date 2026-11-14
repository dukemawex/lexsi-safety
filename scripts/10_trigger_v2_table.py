#!/usr/bin/env python3
"""Table 8: base vs submitted CAST (v1) vs CAST-v2 (wrapper-aware trigger), per prompt form and language."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/hfjob")
jl = lambda p: [json.loads(x) for x in open(p, encoding="utf-8")]  # noqa: E731
base = jl(R / "run/baseline/records.jsonl") + jl(R / "run4/base/records.jsonl")
v1 = {d["sample_id"]: d for d in jl(R / "run2/after/records.jsonl") + jl(R / "run4/cast/records.jsonl")}
v2 = {d["sample_id"]: d for d in jl(R / "run5/v2_plain/records.jsonl") + jl(R / "run5/v2_wrapped/records.jsonl")}
form = lambda i: i.rsplit("-", 1)[1] if i.endswith(("-en_frame", "-ask_local")) else "plain"  # noqa: E731
H = {"harm", "jailbreak"}


def cell(rows, src):
    h = [src[r["sample_id"]] for r in rows if r["category"] in H]
    b = [src[r["sample_id"]] for r in rows if r["category"] == "benign"]
    return f"{100 * sum(x['complied'] for x in h) / len(h):.1f}% / {100 * sum(x['refused'] for x in b) / len(b):.1f}%"


B = {d["sample_id"]: d for d in base}
lines = ["| Prompt form | Language | Base | CAST v1 (submitted) | CAST-v2 (wrapper-aware gate) |", "|---|---|---|---|---|"]
for f in ("plain", "en_frame", "ask_local"):
    for lang in ("eng", "hau", "ibo", "yor"):
        rows = [d for d in base if form(d["sample_id"]) == f and d["language"] == lang]
        if rows:
            lines.append(f"| {f} | {lang} | {cell(rows, B)} | {cell(rows, v1)} | {cell(rows, v2)} |")
lines.append(f"| **pooled** | all | {cell(base, B)} | {cell(base, v1)} | {cell(base, v2)} |")
print("\n".join(lines))
