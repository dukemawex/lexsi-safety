#!/usr/bin/env python3
"""Stage 6: Flatten before/after metrics into a submission-ready table.

Also sketches an optional ROI comparison vs a closed model, using a realistic
sample volume, GPU hours, per-hour GPU pricing, and estimated tokens/sec.
No numbers are invented here; fill them from actual runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--after-dir", type=Path, default=Path("artifacts/after"))
    p.add_argument("--output", type=Path, default=Path("artifacts/metrics/before_after.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    template = {
        "metrics": ["harmful_compliance_rate", "over_refusal_rate", "comprehension_or_engagement_rate", "task_capability"],
        "languages": ["eng", "hau", "ibo", "yor"],
        "before": {},
        "after": {},
        "roi_optional": {
            "closed_model": "fill in only from a real quote or public price",
            "sample_volume": None,
            "gpu_hours": None,
            "gpu_price_per_hour": None,
            "tokens_per_sec": None,
        },
        "sources": [str(args.baseline_dir), str(args.after_dir)],
    }
    print(json.dumps(template, indent=2))
    if args.dry_run:
        print("\nDry run only. Template printed, nothing written.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, indent=2) + "\n")
    print(f"Wrote template to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
