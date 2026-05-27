import argparse
import csv
import os
import statistics
import time
from typing import List, Tuple

import numpy as np
from bioio_ome_zarr.reader import Reader

# Fixed seed so the same URL always exercises the same regions across runs.
SEED = 42
DEFAULT_OUTPUT = "/allen/aics/users/brian.whitney/shard_benchmark/read_benchmark_results.csv"
CSV_FIELDS = ["url", "trial", "elapsed_s", "bytes_read", "throughput_mbs", "shape", "chunks"]


def _sample_slices(
    shape: Tuple[int, ...], n: int, max_elements: int
) -> List[Tuple[slice, ...]]:
    """Return n random sub-volume slices, each touching at most max_elements voxels.

    Uses rejection sampling: draws a random slice across every axis and discards it
    if the product of its extents exceeds the budget.
    """
    rng = np.random.default_rng(SEED)
    slices = []
    while len(slices) < n:
        idx = []
        elements = 1
        for d in shape:
            a = int(rng.integers(0, d))
            b = int(rng.integers(a + 1, d + 1))
            idx.append(slice(a, b))
            elements *= b - a
        if elements <= max_elements:
            slices.append(tuple(idx))
    return slices


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark random-access read throughput for an OME-Zarr image."
    )
    p.add_argument("url", help="OME-Zarr store URL (local path or remote)")
    p.add_argument("--n", type=int, default=20, help="number of random read trials")
    p.add_argument("--max-mb", type=int, default=512, help="max MiB per read trial")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV file to append results to")
    args = p.parse_args()

    reader = Reader(args.url)
    axes = reader.dims.order
    shape = tuple(int(getattr(reader.dims, ax)) for ax in axes)
    # Convert the MiB budget to an element count so rejection sampling is dtype-aware.
    itemsize = np.dtype(reader.dtype).itemsize
    max_elements = args.max_mb * 1024**2 // itemsize

    slices = _sample_slices(shape, args.n, max_elements)

    data = reader.get_image_dask_data()

    times = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_header = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, idx in enumerate(slices):
            t0 = time.perf_counter()
            result = data[idx].compute()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            tput = result.nbytes / elapsed / 1024**2
            print(f"  [{i + 1:>2}/{args.n}]  {elapsed:.3f}s  {result.nbytes / 1024**2:.1f} MiB  {tput:.1f} MiB/s")
            writer.writerow({
                "url": args.url,
                "trial": i,
                "elapsed_s": round(elapsed, 6),
                "bytes_read": result.nbytes,
                "throughput_mbs": round(tput, 3),
                "shape": str(shape),
                "chunks": str(data.chunksize),
            })

    print(f"\nmedian {statistics.median(times):.3f}s  min {min(times):.3f}s  max {max(times):.3f}s")
    print(f"results appended to: {args.output}")


if __name__ == "__main__":
    main()
