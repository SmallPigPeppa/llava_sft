#!/usr/bin/env bash
set -euo pipefail

# Optional: export WANDB_API_KEY=...
python train.py --config configs/demo2k.yaml "$@"
