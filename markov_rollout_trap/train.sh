#!/usr/bin/env bash
set -euo pipefail

SEED=42
DATA_DIR="$(dirname "$0")/data"
CFG="$(dirname "$0")/config.yaml"

for MODE in clean noisy; do
  for MODEL in gmm markov latent; do
    echo "[train] model=${MODEL} mode=${MODE}"
    python -m markov_rollout_trap.train \
      --config "$CFG" \
      --model "$MODEL" \
      --seed "$SEED" \
      --data "$DATA_DIR/${MODE}.parquet" \
      --out_dir "$(dirname "$0")/runs/${MODEL}/${MODE}"
  done
done
