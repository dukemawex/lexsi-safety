#!/usr/bin/env bash
# Run 5: (a) CAST-v2 stronger trigger refit on plain+wrapped calibrate prompts; (b) CircuitKIT EAP-IG pass. See docs/PREREG_RUN5.md.
set -uo pipefail
RUN_REPO="${RUN_REPO:-duke01/lexsi-tiny-aya-earth-safety-run}"
pip install -q --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q --no-cache-dir "transformers>=5.15,<6" datasets accelerate pyyaml "auditkit==1.1.7" "curatorkit[connectors]==0.1.24" "safetune==0.1.16" huggingface_hub uv
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("duke01/lexsi-tiny-aya-earth-safety-run", repo_type="dataset",
                  allow_patterns=["code/*", "run2/eval_set.jsonl", "run/baseline/records.jsonl", "run2/after/records.jsonl",
                                  "run4/base/records.jsonl", "run4/cast/records.jsonl"], local_dir="/work")
PY
cd /work/code
python scripts/run_trigger_v2.py --model CohereLabs/tiny-aya-earth --run-dir /work --output-dir /work/out5 \
  --batch-size 16 --max-new-tokens 128 --upload-repo "$RUN_REPO" --upload-path run5 || echo "TRIGGER_V2_FAILED"
python -c "import torch,gc; print('gpu free', torch.cuda.mem_get_info())"
# CircuitKIT in its own env (pins huggingface_hub<2, transformer-lens 3.8)
uv venv -q /ck --python 3.12 && VIRTUAL_ENV=/ck uv pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 \
  && VIRTUAL_ENV=/ck uv pip install -q "circuitkit==0.2.2" datasets \
  && timeout 1500 /ck/bin/python scripts/run_circuitkit.py --model CohereLabs/tiny-aya-earth --splits /work/run2/eval_set.jsonl \
       --output-dir /work/out5ck --n 24 --dtype float16 2>&1 | grep -v -E "Pruned [0-9]+/|HyperparameterWarning|config = load_and" \
  || echo "CIRCUITKIT_FAILED_OR_TIMEOUT"
python - <<'PY'
import os
from huggingface_hub import HfApi
if os.path.isdir("/work/out5ck"):
    HfApi().upload_folder(folder_path="/work/out5ck", path_in_repo="run5/circuitkit", repo_id=os.environ.get("RUN_REPO", "duke01/lexsi-tiny-aya-earth-safety-run"), repo_type="dataset")
    print("uploaded circuitkit outputs")
PY
echo RUN5_DONE
