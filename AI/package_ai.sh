#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <random|greedy|mcts> [output_path_or_dir]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="$1"
OUTPUT_ARG="${2:-}"
SOURCE_AI=""
ARCHIVE_NAME=""
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

assemble_layout() {
  local output_dir="$1"

  mkdir -p "$output_dir"
  cp "${REPO_ROOT}/AI/main.py" "$output_dir/main.py"
  cp "${REPO_ROOT}/AI/protocol.py" "$output_dir/protocol.py"
  cp "${REPO_ROOT}/AI/common.py" "$output_dir/common.py"
  cp "$SOURCE_AI" "$output_dir/ai.py"

  if [[ ${#EXTRA_FILES[@]} -gt 0 ]]; then
    for extra in "${EXTRA_FILES[@]}"; do
      cp "$extra" "$output_dir/$(basename "$extra")"
    done
  fi

  copy_tree "${REPO_ROOT}/SDK" "${output_dir}/SDK"
  copy_tree "${REPO_ROOT}/tools" "${output_dir}/tools"
}

require_empty_dir() {
  local dir_path="$1"

  if [[ -e "$dir_path" && ! -d "$dir_path" ]]; then
    echo "output path exists and is not a directory: ${dir_path}" >&2
    exit 1
  fi

  mkdir -p "$dir_path"
  if find "$dir_path" -mindepth 1 -print -quit | grep -q .; then
    echo "output directory must be empty: ${dir_path}" >&2
    exit 1
  fi
}

case "$TARGET" in
  random)
    SOURCE_AI="${REPO_ROOT}/AI/ai_random.py"
    ARCHIVE_NAME="ai_rand.zip"
    EXTRA_FILES=()
    ;;
  greedy)
    SOURCE_AI="${REPO_ROOT}/AI/ai_greedy.py"
    ARCHIVE_NAME="ai_greedy.zip"
    EXTRA_FILES=("${REPO_ROOT}/AI/greedy_runtime.py")
    ;;
  mcts)
    SOURCE_AI="${REPO_ROOT}/AI/ai_mcts.py"
    ARCHIVE_NAME="ai_mcts.zip"
    EXTRA_FILES=("${REPO_ROOT}/AI/ai_greedy.py" "${REPO_ROOT}/AI/greedy_runtime.py")
    ;;
  *)
    echo "unknown target: ${TARGET}" >&2
    exit 1
    ;;
esac

if [[ -n "$OUTPUT_ARG" && "$OUTPUT_ARG" != *.zip ]]; then
  OUTPUT_DIR="$OUTPUT_ARG"
  require_empty_dir "$OUTPUT_DIR"
  assemble_layout "$OUTPUT_DIR"
  printf '%s\n' "$OUTPUT_DIR"
  exit 0
fi

OUTPUT_ZIP="${OUTPUT_ARG:-${SCRIPT_DIR}/${ARCHIVE_NAME}}"
OUTPUT_PARENT="$(dirname "$OUTPUT_ZIP")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_ZIP="$(cd "$OUTPUT_PARENT" && pwd)/$(basename "$OUTPUT_ZIP")"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agent-tradition-${TARGET}.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT

assemble_layout "$STAGING_DIR"
rm -f "$OUTPUT_ZIP"
(
  cd "$STAGING_DIR"
  zip -qr "$OUTPUT_ZIP" .
)

printf '%s\n' "$OUTPUT_ZIP"
