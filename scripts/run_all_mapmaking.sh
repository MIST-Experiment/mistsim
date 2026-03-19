#!/usr/bin/env bash
# Run all mapmaking configs sequentially in a tmux-friendly way.
# Usage: just launch in tmux — it picks up where it left off if interrupted.

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
    mars-nv
    mars-nv-mono
    mars-nv-alma
    mars-nv-4beam
    all-sites
    test-compare-old
    mars-nv-multifreq
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
