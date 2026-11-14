#!/usr/bin/env python3
"""CPU smoke test for the local Lexsi scaffold.

Imports AuditKIT and CuratorKIT, runs a tiny echo evaluation, and constructs
a Curator over the benign fixture (clean + exact dedup, no LLM, no network).
SafeTune and CircuitKIT are optional: they need torch and transformers 5.15+
and are not installed on this CPU box by default.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "benign_smoke.jsonl"


def _ok(name: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"PASS  {name}{suffix}")


def _fail(name: str, err: BaseException) -> None:
    print(f"FAIL  {name}: {type(err).__name__}: {err}", file=sys.stderr)
    traceback.print_exc()


def _skip(name: str, reason: str) -> None:
    print(f"SKIP  {name}: {reason}")


def import_optional(modname: str):
    try:
        return __import__(modname)
    except Exception as exc:  # noqa: BLE001 - smoke test reports any import issue
        return exc


def test_auditkit() -> bool:
    import auditkit as ak

    _ok("import auditkit", ak.__version__)
    samples = [
        ak.Sample(input="What is 2+2?", target="4"),
        ak.Sample(input="What is 3+3?", target="6"),
    ]
    result = ak.evaluate(samples, model=lambda prompts: ["4", "6"])
    headline = getattr(result, "headline", None) or {}
    _ok("auditkit.evaluate echo", f"headline={headline}")
    return True


def test_curatorkit() -> bool:
    from curatorkit import Curator, CuratorConfig

    _ok("import curatorkit")
    if not FIXTURE.is_file():
        raise FileNotFoundError(FIXTURE)
    with tempfile.TemporaryDirectory(prefix="curator-smoke-") as tmp:
        result = Curator(
            CuratorConfig(
                dataset=str(FIXTURE),
                clean=True,
                dedup="exact",
                export_formats=["alpaca"],
                output_dir=tmp,
            )
        ).run()
        files = sorted(Path(tmp).name for Path_ in [Path(tmp)])
        out_files = [p.name for p in Path(tmp).iterdir()]
        _ok(
            "curatorkit clean+dedup on benign fixture",
            f"files={out_files} summary={getattr(result, 'summary', lambda: result)() if False else 'ok'}",
        )
        # Print a short non-secret summary if the API provides one.
        if hasattr(result, "print_summary"):
            result.print_summary()
    return True


def test_optional_gpu_kits() -> None:
    for name in ("safetune", "circuitkit"):
        mod = import_optional(name)
        if isinstance(mod, Exception):
            _skip(
                f"import {name}",
                "not installed on this CPU box (needs torch and transformers>=5.15); "
                "install from requirements.txt on a Colab/Kaggle/ZeroGPU runtime later",
            )
        else:
            version = getattr(mod, "__version__", "unknown")
            _ok(f"import {name}", version)


def main() -> int:
    print("lexsi-safety CPU smoke test")
    print(f"python {sys.version.split()[0]}  root={ROOT}")
    failed = 0
    for fn in (test_auditkit, test_curatorkit):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _fail(fn.__name__, exc)
            failed += 1
    test_optional_gpu_kits()
    if failed:
        print(f"\nSMOKE TEST FAILED ({failed} required check(s))")
        return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
