#!/usr/bin/env bash
# Run 3: Belebele capability retention + ungated ablation, from the saved run-2 CAST artifact.
set -euo pipefail
RUN_REPO="${RUN_REPO:-duke01/lexsi-tiny-aya-earth-safety-run}"
pip install -q --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q --no-cache-dir "transformers>=5.15,<6" datasets accelerate pyyaml "auditkit==1.1.7" "curatorkit[connectors]==0.1.24" "safetune==0.1.16" huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("duke01/lexsi-tiny-aya-earth-safety-run", repo_type="dataset", allow_patterns=["code/*", "run2/repair/*"], local_dir="/work")
PY
cd /work/code
python scripts/run_retention.py --model CohereLabs/tiny-aya-earth --repair-dir /work/run2/repair --output-dir /work/out3 \
  --belebele-n 200 --batch-size 8 --max-new-tokens 128 --upload-repo "$RUN_REPO" --upload-path run3
