#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_mcts"
ARCHIVE_NAME="ai_mcts.zip"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/selfplay/mcts_value.npz}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "missing model: ${MODEL_PATH}" >&2
  echo "train first: bash SDK/train_mcts.sh" >&2
  exit 1
fi

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/AI/selfplay" "${BUILD_DIR}/SDK"

cp "${SCRIPT_DIR}/ai_main.py" "${BUILD_DIR}/main.py"
cp "${SCRIPT_DIR}/ai_mcts.py" "${BUILD_DIR}/ai.py"
cp "${SCRIPT_DIR}/ai_greedy.py" "${BUILD_DIR}/AI/ai_greedy.py"
cp "${SCRIPT_DIR}/ai_handcraft.py" "${BUILD_DIR}/AI/ai_handcraft.py"
cp "${MODEL_PATH}" "${BUILD_DIR}/AI/selfplay/mcts_value.npz"

cp "${ROOT_DIR}/SDK/mcts_agent.py" "${BUILD_DIR}/SDK/mcts_agent.py"
cp "${ROOT_DIR}/SDK/mcts_features.py" "${BUILD_DIR}/SDK/mcts_features.py"
cp "${ROOT_DIR}/SDK/training_base.py" "${BUILD_DIR}/SDK/training_base.py"
printf "%s\n" "__all__ = []" > "${BUILD_DIR}/SDK/__init__.py"

cp -R "${ROOT_DIR}/logic" "${BUILD_DIR}/logic"
find "${BUILD_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

rm -f "${SCRIPT_DIR}/${ARCHIVE_NAME}"
cd "${BUILD_DIR}"
zip -r "../${ARCHIVE_NAME}" main.py ai.py logic AI SDK
cd "${SCRIPT_DIR}"
rm -rf "${BUILD_DIR}"

echo "packed: ${SCRIPT_DIR}/${ARCHIVE_NAME}"
