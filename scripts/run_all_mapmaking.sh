#!/usr/bin/env bash
# Run all mapmaking configs sequentially in a tmux-friendly way.
# Usage: just launch in tmux — it picks up where it left off if interrupted.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-data/synthetic_data}"

RUNS=(
    # 40 MHz nominal (7 runs, includes lake beam)
    mars
    mars-nv
    mars-lake
    mars-nv-lake
    mars-nv-lake-alma
    alma-torres
    all-nominal
    # 25 MHz nominal (5 runs, no lake beam)
    mars-25
    mars-nv-25
    mars-nv-alma-25
    alma-torres-25
    all-nolake-25
)

DONEFILE="scripts/.mapmaking_done"
touch "$DONEFILE"

for run in "${RUNS[@]}"; do
    if grep -qxF "$run" "$DONEFILE"; then
        echo "=== Skipping $run (already done) ==="
        continue
    fi
    echo "=== Starting $run at $(date) ==="
    uv run python scripts/run_mapmaking.py "$run" --data-dir "$DATA_DIR"
    echo "$run" >> "$DONEFILE"
    echo "=== Finished $run at $(date) ==="
done

echo "=== All runs complete at $(date) ==="
