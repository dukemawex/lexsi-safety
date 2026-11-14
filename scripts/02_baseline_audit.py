#!/usr/bin/env python3
"""Deprecated entry point. Use scripts/run_baseline.py."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep this file so earlier docs still work.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
