"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""
import pathlib

import bioio_base.benchmark

import bioio_ome_zarr


test_resources_dir = pathlib.Path(__file__).parent.parent / "bioio_ome_zarr" / "tests" / "resources"
bioio_base.benchmark.benchmark(bioio_ome_zarr.reader.Reader, test_resources_dir)
