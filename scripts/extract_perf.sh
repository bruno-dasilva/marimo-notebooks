#!/bin/bash
# Batch-runs demo-parser extract-perf over every .sdfz under
# data/bar_replays/demos/ and writes a gzipped sibling .ndjson.gz under
# data/bar_replays/perf/. Skips outputs already newer than both the
# source .sdfz and the compiled parser, so reruns are cheap and a
# parser rebuild auto-triggers full re-extraction.
#
# Filename convention (preserved between .sdfz and .ndjson.gz):
#   <YYYY-MM-DD_HH-MM-SS-mmm>_<map>_<YYYY.MM.DD>.sdfz
#   leading datetime = replay start; trailing YYYY.MM.DD = engine version.
#
# Cleans up legacy uncompressed `.ndjson` siblings when the corresponding
# `.ndjson.gz` is successfully produced.
#
# Parallelism via xargs -P. Defaults to half of hw.ncpu (extraction is
# CPU-bound parsing + node startup; too many workers thrash). Override:
#   JOBS=8 ./scripts/extract_perf.sh

set -uo pipefail

DEMOS="/Users/bruno/www/marimo-notebooks/data/bar_replays/demos"
PERF="/Users/bruno/www/marimo-notebooks/data/bar_replays/perf"
EXTRACT="/Users/bruno/www/demo-parser/dist/bin/extract-perf.js"
LOG="$PERF/_extract.log"
ERR="$PERF/_extract.err"

if [ -z "${JOBS:-}" ]; then
    if command -v sysctl >/dev/null 2>&1; then
        ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        JOBS=$((ncpu / 2))
        [ "$JOBS" -lt 1 ] && JOBS=1
    elif command -v nproc >/dev/null 2>&1; then
        JOBS=$(($(nproc) / 2))
        [ "$JOBS" -lt 1 ] && JOBS=1
    else
        JOBS=4
    fi
fi

mkdir -p "$PERF"

if [ ! -f "$EXTRACT" ]; then
    echo "Compiled parser not found at $EXTRACT — run \`npm run build\` in demo-parser first." >&2
    exit 1
fi

# Worker function: process exactly one .sdfz. Emits a single line to stdout
# of the form "[HH:MM:SS] STATUS basename" so the user sees live progress.
# The status word is also used to tally counts at the end. Short writes are
# atomic on POSIX (under PIPE_BUF) so concurrent workers don't interleave.
process_one() {
    local f="$1"
    local base out ts
    base=$(basename "$f" .sdfz)
    out="$PERF/$base.ndjson.gz"
    ts=$(date +%H:%M:%S)

    if [ -f "$out" ] && [ "$out" -nt "$f" ] && [ "$out" -nt "$EXTRACT" ]; then
        printf '[%s] SKIP %s\n' "$ts" "$base"
        return 0
    fi

    if node "$EXTRACT" "$f" 2>>"$ERR" | gzip -c > "$out.tmp"; then
        mv "$out.tmp" "$out"
        rm -f "$PERF/$base.ndjson"
        printf '[%s] DONE %s\n' "$ts" "$base"
    else
        rm -f "$out.tmp"
        printf '[%s] FAIL %s\n' "$ts" "$base"
    fi
}
export -f process_one
export PERF EXTRACT ERR

ts=$(date +%H:%M:%S)
printf '[%s] starting (parallelism=%d)\n' "$ts" "$JOBS" | tee -a "$LOG"

status_file=$(mktemp -t extract_status.XXXXXX)
trap 'rm -f "$status_file"' EXIT

# `find -print0 | sort -z` dispatches files to xargs in chronological order
# (filenames start with the replay datetime). Workers run in parallel so
# completions may still interleave slightly when one replay parses faster
# than another, but every job is *started* in order.
#
# `tee -a` shows each completion line on the script's stdout (live progress)
# AND mirrors to the persistent log; `status_file` captures the same stream
# for the final tally.
find "$DEMOS" -maxdepth 1 -name '*.sdfz' -type f -print0 \
    | sort -z \
    | xargs -0 -n 1 -P "$JOBS" bash -c 'process_one "$0"' \
    | tee -a "$LOG" "$status_file"

total=$(awk 'END {print NR+0}' "$status_file")
done_=$(awk '/] DONE / {n++} END {print n+0}' "$status_file")
skipped=$(awk '/] SKIP / {n++} END {print n+0}' "$status_file")
failed=$(awk '/] FAIL / {n++} END {print n+0}' "$status_file")

ts=$(date +%H:%M:%S)
printf '[%s] FINISHED total=%d extracted=%d skipped=%d failed=%d (parallelism=%d)\n' \
    "$ts" "$total" "$done_" "$skipped" "$failed" "$JOBS" | tee -a "$LOG"
