#!/usr/bin/env bash
# Run 2: reuse run-1 baseline generations, sweep gate threshold x alpha on calibrate, eval chosen + naive CAST.
set -euo pipefail
RUN_REPO="${RUN_REPO:-duke01/lexsi-tiny-aya-earth-safety-run}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
pip install -q --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q --no-cache-dir "transformers>=5.15,<6" datasets accelerate pyyaml "auditkit==1.1.7" "curatorkit[connectors]==0.1.24" "safetune==0.1.16" huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("duke01/lexsi-tiny-aya-earth-safety-run", repo_type="dataset", allow_patterns=["code/*", "run/baseline/*"], local_dir="/work")
PY
cd /work/code
python scripts/run_pipeline.py --model CohereLabs/tiny-aya-earth --output-dir /work/out2 \
  --baseline-from /work/run/baseline --batch-size 16 --max-new-tokens 128 \
  --gate-fprs 0.1 0.2 --alphas 0.25 0.5 1.0 --naive-eval-alpha 0.5 \
  --time-budget-min 150 --upload-repo "$RUN_REPO" --upload-path run2
