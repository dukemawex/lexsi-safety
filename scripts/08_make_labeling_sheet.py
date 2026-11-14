#!/usr/bin/env python3
"""Sample a balanced sheet for native-speaker labelling of the refusal/compliance judge.

Writes the sheet WITH prompt and reply text to a path outside git (default results/hfjob/labeling_sheet.csv,
which is gitignored) and a header-only template to docs/labeling_sheet_template.csv.
Labels to fill: human_refused (y/n), human_harmful_content (y/n), human_language_ok (y/n), notes.
"""
import argparse
import csv
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", type=Path, default=Path("results/hfjob/run/baseline/records.jsonl"))
ap.add_argument("--after", type=Path, default=Path("results/hfjob/run2/after/records.jsonl"))
ap.add_argument("--per-language", type=int, default=30)
ap.add_argument("--out", type=Path, default=Path("results/hfjob/labeling_sheet.csv"))
a = ap.parse_args()
B = {json.loads(l)["sample_id"]: json.loads(l) for l in a.baseline.open(encoding="utf-8")}
A = {json.loads(l)["sample_id"]: json.loads(l) for l in a.after.open(encoding="utf-8")}
rnd = random.Random(7)
fields = ["row_id", "sample_id", "language", "category", "condition", "english_source", "prompt", "reply",
          "auto_refused", "auto_complied", "human_refused", "human_harmful_content", "human_language_ok", "notes"]
rows = []
for lang in ["eng", "hau", "ibo", "yor"]:
    ids = [i for i in B if B[i]["language"] == lang]
    rnd.shuffle(ids)
    for i in ids[: a.per_language]:
        cond, src = (("after", A) if rnd.random() < 0.5 else ("baseline", B))
        r = src[i]
        rows.append({"sample_id": i, "language": lang, "category": r["category"], "condition": cond,
                     "english_source": r["english_source"], "prompt": r["prompt"], "reply": r["output"],
                     "auto_refused": r["refused"], "auto_complied": r["complied"],
                     "human_refused": "", "human_harmful_content": "", "human_language_ok": "", "notes": ""})
rnd.shuffle(rows)
for k, r in enumerate(rows):
    r["row_id"] = k + 1
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
Path("docs").mkdir(exist_ok=True)
with open("docs/labeling_sheet_template.csv", "w", newline="") as fh:
    csv.writer(fh).writerow(fields)
print(f"wrote {len(rows)} rows to {a.out}")
