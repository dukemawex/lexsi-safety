#!/usr/bin/env python3
"""Stage 5: AuditKIT after-repair eval on the same held-out splits as stage 2.

Writes a before/after comparison with auditkit.compare. More than one metric
is required by the hackathon rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--repaired-model", default="artifacts/repaired/tiny-aya-earth-steer")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/after"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan = {
        "baseline_dir": str(args.baseline_dir),
        "repaired_model": args.repaired_model,
        "output_dir": str(args.output_dir),
        "api": "auditkit.evaluate(...) then auditkit.compare(base, repaired)",
        "metrics": [
            "harmful_compliance_rate",
            "over_refusal_rate",
            "comprehension_or_engagement_rate",
            "task_capability",
        ],
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("\nDry run only.")
        return 0
    print("Live after-audit needs the baseline artifacts and a GPU runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
