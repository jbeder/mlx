#!/usr/bin/env bash
set -euo pipefail

SEED=42
DATA_DIR="$(dirname "$0")/data"

mkdir -p "$DATA_DIR"

for MODE in clean noisy; do
  echo "[data] mode=${MODE}"
  python -m markov_rollout_trap.make_data \
    --mode "$MODE" \
    --seed "$SEED" \
    --sources 10 \
    --count 10000 \
    --out "$DATA_DIR/${MODE}.parquet"
done
