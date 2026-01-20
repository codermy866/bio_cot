#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -u training/train_hierarchical.py \
  --ablation full \
  "$@"


