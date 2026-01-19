#!/usr/bin/env bash
set -euo pipefail

SEED=42
DATA_DIR="$(dirname "$0")/data"

for MODE in clean noisy; do
  for MODEL in gmm markov latent; do
    echo "[eval] model=${MODEL} mode=${MODE}"
    python -m markov_rollout_trap.eval \
      --data "$DATA_DIR/${MODE}.parquet" \
      --model "$(dirname "$0")/runs/${MODEL}/${MODE}" \
      --seed "$SEED" \
      --rollout_samples 128
  done
done
