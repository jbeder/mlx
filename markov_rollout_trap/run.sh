#!/usr/bin/env bash
set -euo pipefail

SEED=42
DATA_DIR="$(dirname "$0")/data"
CFG="$(dirname "$0")/config.yaml"

mkdir -p "$DATA_DIR"

for MODE in clean noisy; do
  echo "[data] mode=${MODE}"
  python -m markov_rollout_trap.make_data \
    --mode "$MODE" \
    --seed "$SEED" \
    --sources 10 \
    --count 10000 \
    --data_dir "$DATA_DIR"
done

for MODE in clean noisy; do
  for MODEL in gmm markov latent; do
    echo "[train] model=${MODEL} mode=${MODE}"
    python -m markov_rollout_trap.train \
      --config "$CFG" \
      --model "$MODEL" \
      --seed "$SEED" \
      --data_dir "$DATA_DIR" \
      --mode "$MODE" \
      --out_dir "$(dirname "$0")/runs/${MODEL}/${MODE}"
  done
done

for MODE in clean noisy; do
  for MODEL in gmm markov latent; do
    echo "[eval] model=${MODEL} mode=${MODE}"
    python -m markov_rollout_trap.eval \
      --data_dir "$DATA_DIR" \
      --mode "$MODE" \
      --model "$(dirname "$0")/runs/${MODEL}/${MODE}" \
      --seed "$SEED" \
      --rollout_samples 128
  done
done

echo "==> Done"