#!/usr/bin/env python3
"""Stage 1: CuratorKIT ingestion of public safety eval sources.

Default path (no LLM, no new harmful generation):
  load McGill-NLP/tukabench (harm, benign, culture) for eng/hau/ibo/yor,
  clean and exact/minhash dedup, write a held-out eval split plus a small
  contrast split for steering calibration, with a provenance manifest.

This script is a skeleton. It refuses to invent Nigerian Pidgin or
code-mixed harmful prompts. Code-mixed items, if enabled later, must be
built only from paired TukaBench rows that share an Index.

Requires: curatorkit[connectors] and an optional HF token for Hub access.
Does not download models. Does not publish anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "derived" / "tukabench_curated"

LANGUAGES = ("eng", "hau", "ibo", "yor")
CONFIGS = ("afri-jbb-harm", "afri-jbb-benign", "afri-jbb-culture")
DATASET_ID = "McGill-NLP/tukabench"
LICENSE = "CC-BY-NC-4.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned CuratorKIT calls without loading the Hub dataset.",
    )
    p.add_argument(
        "--include-code-mix",
        action="store_true",
        help="Plan a deterministic EN+target mix from paired TukaBench rows (not implemented in the skeleton).",
    )
    return p.parse_args()


def plan(output_dir: Path, include_code_mix: bool) -> dict:
    return {
        "dataset": DATASET_ID,
        "license": LICENSE,
        "cite": ["Akinode et al. 2026 TukaBench", "Chao et al. 2024 JailbreakBench"],
        "configs": list(CONFIGS),
        "splits": list(LANGUAGES),
        "curatorkit": {
            "clean": True,
            "dedup": "minhash",
            "export_formats": ["alpaca", "sft_messages"],
            "output_split": {"eval": 0.8, "steer_calibrate": 0.2},
            "generation_task": None,
            "note": "No synthetic harmful generation. Curation only.",
        },
        "output_dir": str(output_dir),
        "code_mix": (
            "Deferred in this skeleton. If enabled, mix clauses of paired "
            "Goal and Goal_Translation rows that share Index. Do not author new attacks."
            if include_code_mix
            else "Off."
        ),
        "pidgin": "Out of scope until a licensed public source exists.",
    }


def run_curator(output_dir: Path) -> None:
    from curatorkit import Curator, CuratorConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    # A full Hub run needs the connectors extra. Keep the call explicit so the
    # API is the one we will use on a GPU runtime, not a reimplementation.
    result = Curator(
        CuratorConfig(
            dataset={"name": DATASET_ID, "max_samples": 2400},
            clean=True,
            dedup="minhash",
            export_formats=["alpaca"],
            output_dir=str(output_dir),
        )
    ).run()
    manifest = output_dir / "run_plan.json"
    if hasattr(result, "print_summary"):
        result.print_summary()
    manifest.write_text(json.dumps({"status": "completed"}, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    planned = plan(args.output_dir, args.include_code_mix)
    print(json.dumps(planned, indent=2))
    if args.dry_run:
        print("\nDry run only. No Hub download, no files written.")
        return 0
    print(
        "Live Hub curation is left for a GPU or laptop run after Duke accepts "
        "dataset terms. Re-run without --dry-run once curatorkit[connectors] "
        "is installed and network access is intended.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
