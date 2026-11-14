#!/usr/bin/env python3
"""Stage 4 (optional): CircuitKIT location of refusal-relevant structure.

Tiny Aya is an experimental CircuitKIT target via a TransformerLens 3.8
cohere2 port. Import circuitkit before transformer_lens. The model is gated.

Preferred cheap path: use SafeTune interpret / the refusal direction already
extracted in stage 3, then optionally run a small EAP or EAP-IG discovery
with a low n_examples. Full ACDC is too slow on a 3.35B model for this week.

Skip this stage if a 16GB T4 OOMs. The submission still chains three Track 2
repos without it (CuratorKIT, AuditKIT, SafeTune). CircuitKIT is the preferred
fourth repo when VRAM allows.
"""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="CohereLabs/tiny-aya-earth")
    p.add_argument("--algorithm", default="eap-ig")
    p.add_argument("--n-examples", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan = {
        "model": args.model,
        "api": "import circuitkit as ck; model = ck.load_model(...); ck.discover(model, task=..., algorithm=..., n_examples=...)",
        "algorithm": args.algorithm,
        "n_examples": args.n_examples,
        "import_order": "import circuitkit before transformer_lens",
        "skip_if": "VRAM under about 16GB after the model and TL activations, or parity gate fails",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("\nDry run only. No model load.")
        return 0
    print("Live CircuitKIT discovery needs a GPU runtime and gated-model access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
