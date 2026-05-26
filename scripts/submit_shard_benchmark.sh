#!/bin/bash
# Submit one read-benchmark job per shard size.
# Usage: bash submit_shard_benchmark.sh

# Activate pyenv virtualenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate shard_benchmark

SCRIPT=/allen/aics/users/brian.whitney/Desktop/Repos/bioio-ome-zarr/scripts/read_benchmark.py
BASE=https://vast-files.int.allencell.org/users/brian.whitney/shard_benchmark/3500008271_20260227_20X_Timelapse
OUTPUT=/allen/aics/users/brian.whitney/shard_benchmark/read_benchmark_results.csv
LOGS=/allen/aics/users/brian.whitney/shard_benchmark/logs

mkdir -p "$LOGS"

for MB in 250 500 1024 2048; do
    URL="${BASE}_shard${MB}mb.ome.zarr"
    sbatch \
        --job-name="read_bench_${MB}mb" \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=2:00:00 \
        --output="${LOGS}/%j_read_bench_${MB}mb.out" \
        --error="${LOGS}/%j_read_bench_${MB}mb.err" \
        --wrap="python3 $SCRIPT $URL --output $OUTPUT"
    echo "Submitted: shard ${MB}mb"
done
