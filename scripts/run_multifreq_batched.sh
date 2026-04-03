#!/bin/bash
# Run all-nominal-multifreq in batches of 21 frequencies.
# Each batch saves a temporary npz; a final step merges and cleans up.

set -e

cd "$(dirname "$0")/.."

BATCH_SIZE=21
TOTAL=101  # indices 0..100 (25-125 MHz)
RUN=all-nominal-multifreq
DATA_DIR="${1:-data/synthetic_data}"
RESULTS_DIR=notebooks/mapmaking/results

mkdir -p "$RESULTS_DIR"

batch=0
for start in $(seq 0 $BATCH_SIZE $((TOTAL - 1))); do
    end=$((start + BATCH_SIZE - 1))
    if [ $end -ge $TOTAL ]; then
        end=$((TOTAL - 1))
    fi

    indices=$(seq $start $end)
    echo "=== Batch $batch: freq indices $start..$end ==="

    uv run python scripts/run_mapmaking.py "$RUN" \
        --data-dir "$DATA_DIR" \
        --freq-indices $indices \
        --no-notebook \
        --output-dir "$RESULTS_DIR"

    mv "$RESULTS_DIR/${RUN}.npz" "$RESULTS_DIR/${RUN}_batch${batch}.npz"

    echo "=== Batch $batch done ==="
    batch=$((batch + 1))
done

echo ""
echo "=== All batches complete. Merging... ==="

uv run python -c "
import numpy as np
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
files = sorted(results_dir.glob('${RUN}_batch*.npz'))
print(f'Merging {len(files)} batch files...')

batches = [np.load(f) for f in files]

merged = {}
for key in batches[0].files:
    arrs = [b[key] for b in batches]
    if key in ('lmax', 'lmax_sim', 'config_yaml', 'multi_freq', 'freqs'):
        merged[key] = arrs[0]
    else:
        merged[key] = np.concatenate(arrs, axis=0)

merged['multi_freq'] = True
out = results_dir / '${RUN}.npz'
np.savez(out, **merged)
print(f'Saved merged results to {out} with {len(merged[\"sim_freqs\"])} frequencies')

for b in batches:
    b.close()
for f in files:
    f.unlink()
print(f'Removed {len(files)} batch files')
"

echo "=== Done! ==="
