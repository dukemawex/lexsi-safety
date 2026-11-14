# Hugging Face Jobs runs (what was actually run, 14 Nov 2026)

All jobs: `--flavor t4-medium` (1x T4 16 GB, 30 GB RAM, $0.60/hour), image `python:3.12`, token passed with `--secrets HF_TOKEN` (read from the local environment, never printed). Code and outputs go through the PRIVATE dataset repo `duke01/lexsi-tiny-aya-earth-safety-run` (`code/`, `run/`, `run2/`, `run3/`, `run4/`, `run5/`, `run6/`).

```bash
hf jobs run --detach --name <name> --flavor t4-medium --timeout <cap> --secrets HF_TOKEN python:3.12 -- \
  bash -c 'pip install -q huggingface_hub && hf download duke01/lexsi-tiny-aya-earth-safety-run --repo-type dataset --include "code/*" --local-dir /work && bash /work/code/scripts/<entry>.sh'
```

| Run | Job id | Entry | Cap | What it did | Billed time |
|---|---|---|---|---|---|
| 1 | 6af84dd923985fb9cd82011a | `job_entry.sh` | 3h | CuratorKIT + baseline on 957 eval prompts + CAST fit; first alpha sweep showed the default gate over-fires; cancelled to redesign the sweep | about 20 min |
| 2 | 6af8535629dd05ff036738dc | `job_entry_v2.sh` | 2h | Reused run-1 baseline; CAST fit; gate budget x alpha sweep on calibration; eval with chosen repair; naive default-gate comparison | 43.3 min |
| 3 | 6af85b6a29dd05ff036742e2 | `job_entry_v3.sh` | 75m | Belebele retention; failed with CUDA OOM in the gate prefill (full-vocab logits on long passages) | 6.1 min |
| 3b | 6af85d7a23985fb9cd821457 | `job_entry_v3.sh` (fixed: `logits_to_keep=1` in the gate, batch 8) | 75m | Belebele retention + ungated ablation | 28.2 min |
| 4 | 6af85df223985fb9cd8214b4 | `job_entry_v4.sh` | 50m | Code-mixed wrapper variants, base vs CAST | 30.2 min |
| 5 | 6af8711323985fb9cd822bec | `job_entry_v5.sh` | 70m | Pre-registered CAST-v2 (wrapper-aware gate) refit + eval on 2,397 plain and wrapped prompts (not adopted); CircuitKIT EAP-IG with full chat template ran out of GPU memory | 28.1 min |
| 6 | 6af8787f23985fb9cd823578 | `job_entry_v6.sh` | 25m | CircuitKIT retry with short chat turns; out of GPU memory again, so CircuitKIT was skipped | 2.3 min |

Monitoring: `hf jobs inspect <id>`, `hf jobs logs -f <id>`, `hf jobs cancel <id>`.

Why Jobs and not a Space: Jobs bill only while starting or running and stop on exit; a GPU Space bills until paused. Same T4 hourly price.

Total: about 9,500 billed seconds (2.6 h) at $0.60/h, about $1.59 (runs 1 to 4: $1.28 of the first ~$2; runs 5 and 6: about $0.31 of the extra ~$1 Duke approved on 14 Nov 15:26 WAT).
