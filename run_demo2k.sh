#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/demo2k.yaml "$@"
