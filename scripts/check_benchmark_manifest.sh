#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_PATH="$ROOT_DIR/benches/BASELINE_MANIFEST.txt"
TMP_PATH="$(mktemp)"
trap 'rm -f "$TMP_PATH"' EXIT

# Extract benchmark baseline descriptors from bench sources.
# One descriptor per line, sorted for deterministic diffs.
if command -v rg >/dev/null 2>&1; then
  rg --no-filename --only-matching 'BASELINE\|[^"\\]+' "$ROOT_DIR"/benches/bench_*.rs \
    | sort -u > "$TMP_PATH"
else
  grep -hoE 'BASELINE\|[^"\\]+' "$ROOT_DIR"/benches/bench_*.rs \
    | sort -u > "$TMP_PATH"
fi

if [[ "${1:-}" == "--write" ]]; then
  cp "$TMP_PATH" "$MANIFEST_PATH"
  echo "Wrote benchmark baseline manifest: $MANIFEST_PATH"
  exit 0
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Benchmark baseline manifest missing: $MANIFEST_PATH" >&2
  echo "Run: scripts/check_benchmark_manifest.sh --write" >&2
  exit 1
fi

if ! diff -u "$MANIFEST_PATH" "$TMP_PATH"; then
  echo "Benchmark baseline manifest drift detected." >&2
  echo "If intentional, update with: scripts/check_benchmark_manifest.sh --write" >&2
  exit 1
fi

echo "Benchmark baseline manifest check passed."
