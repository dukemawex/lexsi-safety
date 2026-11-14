# Scripts

| Script | Stage | Default |
| --- | --- | --- |
| `00_smoke_test.py` | CPU import and benign CuratorKIT run | Runs fully on this box |
| `run_baseline.py` | CuratorKIT + AuditKIT baseline | `--dry-run` or `--model mock` on CPU; GPU via `hf jobs` after approval |
| `01_curate_eval_set.py` | CuratorKIT | `--dry-run` until Hub access is intended |
| `02_baseline_audit.py` | AuditKIT before | `--dry-run` on CPU |
| `03_steer_or_harden.py` | SafeTune | `--dry-run` on CPU |
| `04_locate_circuits.py` | CircuitKIT, optional | `--dry-run` on CPU |
| `05_after_audit.py` | AuditKIT after | `--dry-run` on CPU |
| `06_report_metrics.py` | Metrics table | `--dry-run` prints a template |

All live paths refuse to invent harmful prompts and do not publish to GitHub or Hugging Face.
