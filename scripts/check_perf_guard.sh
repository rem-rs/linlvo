#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_BASELINE_PATH="$ROOT_DIR/benches/PERF_GUARD_BASELINE.txt"
BASELINE_PATH="${PERF_GUARD_BASELINE_PATH:-$DEFAULT_BASELINE_PATH}"
CURRENT_PATH="$ROOT_DIR/benches/PERF_GUARD_CURRENT.txt"
REPORT_PATH="$ROOT_DIR/benches/PERF_GUARD_REPORT.txt"

TOLERANCE="${PERF_GUARD_TOLERANCE:-0.50}"
TOLERANCE_MAP="${PERF_GUARD_TOLERANCE_MAP:-}"
KEYS=(
  "spmv_1d_n5000_p50_ms"
  "spmv_1d_n5000_p95_ms"
  "cg_1d_n1000_p50_ms"
  "cg_1d_n1000_p95_ms"
)

run_current() {
  (cd "$ROOT_DIR" && cargo run --release --example perf_guard --quiet) > "$CURRENT_PATH"
}

extract_value() {
  local key="$1"
  local path="$2"
  awk -F'=' -v k="$key" '$1==k {print $2}' "$path" | head -n1
}

metric_tolerance() {
  local key="$1"
  local entry k v
  IFS=',' read -r -a pairs <<< "$TOLERANCE_MAP"
  for entry in "${pairs[@]}"; do
    entry="${entry//[[:space:]]/}"
    [[ -z "$entry" ]] && continue
    if [[ "$entry" == *=* ]]; then
      k="${entry%%=*}"
      v="${entry#*=}"
      if [[ "$k" == "$key" && -n "$v" ]]; then
        echo "$v"
        return
      fi
    fi
  done
  echo "$TOLERANCE"
}

run_current

if [[ "${1:-}" == "--write" ]]; then
  cp "$CURRENT_PATH" "$BASELINE_PATH"
  cat > "$REPORT_PATH" <<EOF
PERF_GUARD_REPORT
mode=write
baseline=$BASELINE_PATH
source=$CURRENT_PATH
EOF
  echo "Wrote perf guard baseline: $BASELINE_PATH"
  exit 0
fi

if [[ ! -f "$BASELINE_PATH" ]]; then
  echo "Perf guard baseline missing: $BASELINE_PATH" >&2
  echo "Run: scripts/check_perf_guard.sh --write" >&2
  exit 1
fi

regressions=0
{
  echo "PERF_GUARD_REPORT"
  echo "tolerance=$TOLERANCE"
  echo "tolerance_map=$TOLERANCE_MAP"
  echo "baseline=$BASELINE_PATH"
  echo "current=$CURRENT_PATH"

  for key in "${KEYS[@]}"; do
    cur="$(extract_value "$key" "$CURRENT_PATH")"
    base="$(extract_value "$key" "$BASELINE_PATH")"

    if [[ -z "$cur" || -z "$base" ]]; then
      echo "metric=$key status=missing current=$cur baseline=$base"
      regressions=$((regressions + 1))
      continue
    fi

    tol="$(metric_tolerance "$key")"
    limit="$(awk -v b="$base" -v t="$tol" 'BEGIN { printf "%.6f", b * (1.0 + t) }')"
    ratio="$(awk -v c="$cur" -v b="$base" 'BEGIN { if (b == 0) { print "inf" } else { printf "%.4f", c / b } }')"

    if awk -v c="$cur" -v l="$limit" 'BEGIN { exit !(c > l) }'; then
      echo "metric=$key status=regressed tolerance=$tol current_ms=$cur baseline_ms=$base limit_ms=$limit ratio=$ratio"
      echo "::warning::Performance regression on $key: current=$cur ms baseline=$base ms limit=$limit ms tolerance=$tol (ratio=$ratio)"
      regressions=$((regressions + 1))
    else
      echo "metric=$key status=ok tolerance=$tol current_ms=$cur baseline_ms=$base limit_ms=$limit ratio=$ratio"
    fi
  done
} > "$REPORT_PATH"

cat "$REPORT_PATH"

if (( regressions > 0 )); then
  echo "Perf guard detected $regressions regression(s)." >&2
  exit 2
fi

echo "Perf guard check passed."
