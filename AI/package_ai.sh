#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <random|greedy|mcts> [output_dir]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="$1"
OUTPUT_DIR="${2:-$(mktemp -d "${TMPDIR:-/tmp}/agent-tradition-${TARGET}.XXXXXX")}"
SOURCE_AI=""
declare -a EXTRA_FILES=()

copy_tree() {
  local source="$1"
  local target="$2"
  mkdir -p "$target"
  cp -R "$source/." "$target/"
  find "$target" -name '__pycache__' -type d -prune -exec rm -rf {} +
  find "$target" -name '*.pyc' -delete
  find "$target" -name '.DS_Store' -delete
}

case "$TARGET" in
  random)
    SOURCE_AI="${REPO_ROOT}/AI/ai_random.py"
    EXTRA_FILES=()
    ;;
  greedy)
    SOURCE_AI="${REPO_ROOT}/AI/ai_greedy.py"
    EXTRA_FILES=("${REPO_ROOT}/AI/greedy_runtime.py")
    ;;
  mcts)
    SOURCE_AI="${REPO_ROOT}/AI/ai_mcts.py"
    EXTRA_FILES=("${REPO_ROOT}/AI/ai_greedy.py" "${REPO_ROOT}/AI/greedy_runtime.py")
    ;;
  *)
    echo "unknown target: ${TARGET}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"
cp "${REPO_ROOT}/AI/main.py" "$OUTPUT_DIR/main.py"
cp "${REPO_ROOT}/AI/protocol.py" "$OUTPUT_DIR/protocol.py"
cp "${REPO_ROOT}/AI/common.py" "$OUTPUT_DIR/common.py"
cp "$SOURCE_AI" "$OUTPUT_DIR/ai.py"

if [[ ${#EXTRA_FILES[@]} -gt 0 ]]; then
  for extra in "${EXTRA_FILES[@]}"; do
    cp "$extra" "$OUTPUT_DIR/$(basename "$extra")"
  done
fi

copy_tree "${REPO_ROOT}/SDK" "${OUTPUT_DIR}/SDK"
copy_tree "${REPO_ROOT}/tools" "${OUTPUT_DIR}/tools"

printf '%s\n' "$OUTPUT_DIR"
