"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""
import pathlib

import bioio_base.benchmark

import bioio_ome_zarr


benchmark_functions: bioio_base.benchmark.BenchmarkDefinition = [
    {
        "prefix": "Get resolution levels",
        "test": lambda test_file, Reader: Reader(test_file).resolution_levels,
    },
]


# This file is under /scripts while the test resourcess are under /bioio_ome_zarr/tests/resources
test_resources_dir = pathlib.Path(__file__).parent.parent / "bioio_ome_zarr" / "tests" / "resources"
bioio_base.benchmark.benchmark(bioio_ome_zarr.reader.Reader, test_resources_dir, benchmark_functions)
