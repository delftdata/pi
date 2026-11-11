#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-qwen2.5-coder:7b}"
TASK_ID="${2:-}"
CONFIG_FILE="$SCRIPT_DIR/swe-tasks.json"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "=========================================================="
echo " Pi Durable Local SWE Runner"
echo " Model:       $MODEL"
echo " Config:      $CONFIG_FILE"
echo " Results Dir: $RESULTS_DIR"
if [ -n "$TASK_ID" ] && [[ "$TASK_ID" != --* ]]; then
  echo " Filter Task: $TASK_ID"
fi
echo "=========================================================="

RESOLVER_URL="$(node -p 'require("node:url").pathToFileURL(process.argv[1]).href' "$SCRIPT_DIR/packages/coding-agent/src/experimental/source-resolver.ts")"

ARGS=(
  --import "$RESOLVER_URL"
  "$SCRIPT_DIR/scripts/run-swe-durable.ts"
  --config "$CONFIG_FILE"
  --model "$MODEL"
  --resultsDir "$RESULTS_DIR"
)

if [ -n "$TASK_ID" ] && [[ "$TASK_ID" != --* ]]; then
  ARGS+=(--task "$TASK_ID")
  shift 2 || true
  ARGS+=("$@")
elif [ "$#" -gt 1 ]; then
  shift 1
  ARGS+=("$@")
fi

node "${ARGS[@]}"

echo ""
echo "=========================================================="
echo " Recent Results in $RESULTS_DIR:"
echo "=========================================================="
ls -lt "$RESULTS_DIR" | head -n 10
