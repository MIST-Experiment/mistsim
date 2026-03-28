#!/usr/bin/env bash
# Run all mapmaking configs sequentially in a tmux-friendly way.
# Usage: just launch in tmux — it picks up where it left off if interrupted.

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
    mars
    mars-nv
    mars-lake
    mars-nv-lake
    mars-nv-lake-alma
    all-nominal
    alma-torres
)

DONEFILE="scripts/.mapmaking_done"
touch "$DONEFILE"

for run in "${RUNS[@]}"; do
    if grep -qxF "$run" "$DONEFILE"; then
        echo "=== Skipping $run (already done) ==="
        continue
    fi
    echo "=== Starting $run at $(date) ==="
    uv run python scripts/run_mapmaking.py "$run"
    echo "$run" >> "$DONEFILE"
    echo "=== Finished $run at $(date) ==="
done

echo "=== All runs complete at $(date) ==="
