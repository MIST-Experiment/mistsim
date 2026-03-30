#!/bin/bash
# Run all-nominal-multifreq-lake in batches of 21 frequencies to avoid crashes.
# Each batch saves to a separate npz; a final Python step merges them.

set -e

cd "$(dirname "$0")/.."

BATCH_SIZE=21
TOTAL=86  # indices 0..85 (40-125 MHz)
RUN=all-nominal-multifreq-lake
RESULTS_DIR=notebooks/mapmaking/results
BATCH_DIR="${RESULTS_DIR}/batches"

mkdir -p "$BATCH_DIR"

batch=0
for start in $(seq 0 $BATCH_SIZE $((TOTAL - 1))); do
    end=$((start + BATCH_SIZE - 1))
    if [ $end -ge $TOTAL ]; then
        end=$((TOTAL - 1))
    fi

    indices=$(seq $start $end)
    echo "=== Batch $batch: freq indices $start..$end ==="

    uv run python scripts/run_mapmaking.py "$RUN" \
        --freq-indices $indices \
        --no-notebook \
        --output-dir "$BATCH_DIR"

    # Rename to avoid overwrite
    mv "$BATCH_DIR/${RUN}.npz" "$BATCH_DIR/${RUN}_batch${batch}.npz"

    echo "=== Batch $batch done ==="
    batch=$((batch + 1))
done

echo ""
echo "=== All batches complete. Merging... ==="

uv run python -c "
import numpy as np
from pathlib import Path

batch_dir = Path('${BATCH_DIR}')
files = sorted(batch_dir.glob('${RUN}_batch*.npz'))
print(f'Merging {len(files)} batch files...')

# Load all batches
batches = [np.load(f) for f in files]

# Concatenate arrays along freq axis (axis 0 for 2D+, scalar for 1D)
merged = {}
for key in batches[0].files:
    arrs = [b[key] for b in batches]
    if key in ('lmax', 'config_yaml', 'multi_freq'):
        merged[key] = arrs[0]
    elif key == 'freqs':
        merged[key] = arrs[0]  # full freq array is the same in each batch
    else:
        merged[key] = np.concatenate(arrs, axis=0)

merged['multi_freq'] = True
out = Path('${RESULTS_DIR}/${RUN}.npz')
np.savez(out, **merged)
print(f'Saved merged results to {out} with {len(merged[\"sim_freqs\"])} frequencies')

for b in batches:
    b.close()
"

echo "=== Done! ==="
