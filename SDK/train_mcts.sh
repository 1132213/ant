#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_PATH="${SAVE_PATH:-${ROOT_DIR}/AI/selfplay/mcts_value.npz}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/AI/selfplay/mcts_train_latest.log}"
GAMES="${GAMES:-24}"
ROUNDS="${ROUNDS:-80}"
SIMULATIONS="${SIMULATIONS:-8}"
DEPTH="${DEPTH:-3}"
TRAIN_EVERY="${TRAIN_EVERY:-4}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.05}"
L2="${L2:-0.0001}"
OPPONENTS="${OPPONENTS:-self,handcraft}"
SEED="${SEED:-}"

mkdir -p "$(dirname "${SAVE_PATH}")"
mkdir -p "$(dirname "${LOG_PATH}")"

cd "${ROOT_DIR}"

CMD=(
  python SDK/train_mcts_selfplay.py
  --save "${SAVE_PATH}"
  --games "${GAMES}"
  --rounds "${ROUNDS}"
  --simulations "${SIMULATIONS}"
  --depth "${DEPTH}"
  --train_every "${TRAIN_EVERY}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --l2 "${L2}"
  --opponents "${OPPONENTS}"
  --log_every 1
)

if [ -n "${SEED}" ]; then
  CMD+=(--seed "${SEED}")
fi

echo "[train-script] root=${ROOT_DIR}"
echo "[train-script] default_save=${SAVE_PATH}"
echo "[train-script] default_log=${LOG_PATH}"
echo "[train-script] default_opponents=${OPPONENTS}"
echo "[train-script] cmd=${CMD[*]} $*"

PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${CMD[@]}" "$@" | tee "${LOG_PATH}"
