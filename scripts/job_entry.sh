#!/usr/bin/env bash
# Runs inside the HF Job (python:3.12 image). HF_TOKEN arrives as a job secret; never echoed.
set -euo pipefail
RUN_REPO="${RUN_REPO:-duke01/lexsi-tiny-aya-earth-safety-run}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
pip install -q --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q --no-cache-dir "transformers>=5.15,<6" datasets accelerate pyyaml "auditkit==1.1.7" "curatorkit[connectors]==0.1.24" "safetune==0.1.16" huggingface_hub
python -c "import torch,transformers,safetune;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("duke01/lexsi-tiny-aya-earth-safety-run", repo_type="dataset", allow_patterns=["code/*"], local_dir="/work")
PY
cd /work/code
python scripts/run_pipeline.py --model CohereLabs/tiny-aya-earth --output-dir /work/out \
  --batch-size 16 --max-new-tokens 128 --ablation-ungated --time-budget-min 150 --upload-repo "$RUN_REPO"
