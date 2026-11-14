#!/usr/bin/env python3
"""Stage 3: SafeTune repair.

Primary path (cheap, no weight update): RefusalDirectionTrainer.
  Calibrate on a small held-out contrast set of TukaBench harmful vs benign
  prompts, then wrap the frozen Tiny Aya Earth checkpoint.

Fallback if steering over-refuses or collapses capability:
  SafeGradTrainer (train-time harden) on a small LoRA run. Leave max_len
  unset or >= 512 (Tiny Aya chat preamble is about 366 tokens). SafeTune
  docs suggest pick_layer=24 of 36 if the automatic layer sweep is poor.

Do not use Recover/ReSta unless we first create a drifted fine-tune.
ReSta on Tiny Aya wants alpha around 0.25; alpha=1 breaks the model.
"""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="CohereLabs/tiny-aya-earth")
    p.add_argument("--method", choices=["steer", "harden"], default="steer")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--pick-layer", type=int, default=24)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan = {
        "model": args.model,
        "method": args.method,
        "api": (
            "from safetune.runner import steer; "
            "steer.RefusalDirectionTrainer(model, tokenizer, alpha=...).calibrate(harmful=..., harmless=...)"
            if args.method == "steer"
            else "from safetune.runner import harden; harden.SafeGradTrainer(...).train(train_ds, safety_dataset=...)"
        ),
        "alpha": args.alpha,
        "pick_layer": args.pick_layer,
        "colab_note": "SafeTune docs: pip uninstall -y torchao before importing SafeTune on Colab.",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("\nDry run only. No model load, no training.")
        return 0
    print("Live SafeTune repair needs a GPU runtime and the safetune package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
