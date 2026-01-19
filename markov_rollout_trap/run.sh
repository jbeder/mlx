#!/usr/bin/env bash
set -euo pipefail

SEED=42
DATA_DIR="$(dirname "$0")/data"
CFG="$(dirname "$0")/config.yaml"

mkdir -p "$DATA_DIR"

echo "==> Generating data (clean & noisy)"
for MODE in clean noisy; do
  python -m markov_rollout_trap.make_data \
    --mode "$MODE" \
    --seed "$SEED" \
    --sources 10 \
    --count 10000 \
    --out "$DATA_DIR/${MODE}.parquet"
done
echo "==> Training runs (3 models x 2 datasets)"
for MODE in clean noisy; do
  for MODEL in gmm markov latent; do
    python -m markov_rollout_trap.train \
      --config "$CFG" \
      --model "$MODEL" \
      --seed "$SEED" \
      --data "$DATA_DIR/${MODE}.parquet" \
      --out_dir "$(dirname "$0")/runs/${MODEL}/${MODE}"
  done
done
echo "==> Evaluations (each run on its matching dataset)"
for MODE in clean noisy; do
  for MODEL in gmm markov latent; do
    python -m markov_rollout_trap.eval \
      --data "$DATA_DIR/${MODE}.parquet" \
      --model "$(dirname "$0")/runs/${MODEL}/${MODE}" \
      --seed "$SEED"
  done
done

echo "==> Done"