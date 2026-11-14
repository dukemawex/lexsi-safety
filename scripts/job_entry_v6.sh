#!/usr/bin/env bash
# Run 6: CircuitKIT retry with short chat turns (run 5's full chat template OOMed on the T4). See docs/PREREG_RUN5.md.
set -uo pipefail
RUN_REPO="${RUN_REPO:-duke01/lexsi-tiny-aya-earth-safety-run}"
pip install -q uv
uv venv -q /ck --python 3.12
VIRTUAL_ENV=/ck uv pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=/ck uv pip install -q "circuitkit==0.2.2" datasets
/ck/bin/python -c "from huggingface_hub import snapshot_download as s; s('$RUN_REPO', repo_type='dataset', allow_patterns=['code/*','run2/eval_set.jsonl'], local_dir='/work')"
cd /work/code
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 1200 /ck/bin/python scripts/run_circuitkit.py --model CohereLabs/tiny-aya-earth \
  --splits /work/run2/eval_set.jsonl --output-dir /work/out6ck --n 24 --dtype float16 --no-preamble 2>&1 \
  | grep -v -E "Pruned [0-9]+/|HyperparameterWarning|config = load_and|Layer [0-9]+: Pruned" || echo "CIRCUITKIT_FAILED_OR_TIMEOUT"
/ck/bin/python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='/work/out6ck', path_in_repo='run6/circuitkit', repo_id='$RUN_REPO', repo_type='dataset'); print('uploaded')"
echo RUN6_DONE
