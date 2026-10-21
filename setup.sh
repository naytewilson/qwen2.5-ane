#!/bin/bash
# One-command setup for qwen-ane. Swift only — no Python, no coremltools.
#
#   1. Build the engine (zero external dependencies).
#   2. Put the compiled model where the runtime looks for it:
#        - already in models/        → used as is
#        - QWEN_MODEL_PATH=<.mlmodelc> → linked into models/
#        - QWEN_MODEL_URL=<.zip/.tar.gz of the .mlmodelc> → downloaded,
#          unpacked and linked into models/
#   3. Probe the artifact: it must load as a multi-function model
#      (`infer` + batched `prefill`). A single-function artifact still runs
#      but prompt prefill falls back to one ANE dispatch per token, so setup
#      refuses it unless QWEN_ALLOW_SINGLE_FUNCTION=1.
set -euo pipefail
cd "$(dirname "$0")"

DEST_DIR="models/qwen25-1.5b-ane-ctx512-lut4"
MLMODELC="$DEST_DIR/qwen25_monolithic_full_lut4.mlmodelc"
BIN=".build/release/qwen-ane"

echo "==> Building engine"
swift build -c release
[ -x "$BIN" ] || { echo "build produced no $BIN" >&2; exit 1; }

echo "==> Locating model"
mkdir -p "$DEST_DIR"
if [ -e "$MLMODELC" ]; then
    echo "    using $MLMODELC"
elif [ -n "${QWEN_MODEL_PATH:-}" ]; then
    [ -d "$QWEN_MODEL_PATH" ] || { echo "QWEN_MODEL_PATH is not a directory: $QWEN_MODEL_PATH" >&2; exit 1; }
    ln -s "$(cd "$QWEN_MODEL_PATH" && pwd)" "$MLMODELC"
    echo "    linked $QWEN_MODEL_PATH"
elif [ -n "${QWEN_MODEL_URL:-}" ]; then
    CACHE="${HOME}/.cache/qwen-ane/models"
    mkdir -p "$CACHE"
    ARCHIVE="$CACHE/$(basename "${QWEN_MODEL_URL%%\?*}")"
    echo "    downloading $QWEN_MODEL_URL"
    curl -fL --retry 3 -C - -o "$ARCHIVE" "$QWEN_MODEL_URL"
    case "$ARCHIVE" in
        *.zip)            ditto -x -k "$ARCHIVE" "$CACHE" ;;
        *.tar.gz|*.tgz)   tar -xzf "$ARCHIVE" -C "$CACHE" ;;
        *) echo "unsupported archive type: $ARCHIVE" >&2; exit 1 ;;
    esac
    FOUND=$(find "$CACHE" -maxdepth 3 -name 'qwen25_monolithic_full_lut4.mlmodelc' -type d | head -1)
    [ -n "$FOUND" ] || { echo "archive did not contain qwen25_monolithic_full_lut4.mlmodelc" >&2; exit 1; }
    ln -s "$FOUND" "$MLMODELC"
    echo "    installed $FOUND"
else
    cat >&2 <<MSG
No model found. Provide the compiled model one of these ways:
  QWEN_MODEL_PATH=/path/to/qwen25_monolithic_full_lut4.mlmodelc ./setup.sh
  QWEN_MODEL_URL=https://…/qwen25_monolithic_full_lut4.mlmodelc.zip ./setup.sh
MSG
    exit 1
fi

echo "==> Probing artifact"
if "$BIN" --probe-model "$MLMODELC"; then
    echo "==> Ready. Start the server with:  $BIN --serve"
elif [ "${QWEN_ALLOW_SINGLE_FUNCTION:-0}" = "1" ]; then
    echo "==> Single-function artifact accepted (QWEN_ALLOW_SINGLE_FUNCTION=1); prompt prefill will be slow."
else
    echo "Artifact is not multi-function; refusing. Set QWEN_ALLOW_SINGLE_FUNCTION=1 to accept it." >&2
    exit 1
fi
