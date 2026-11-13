"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""
import pathlib

import bioio_base.benchmark

import bioio_ome_zarr


def read_sub_region(test_file: pathlib.Path) -> None:
    """Read the top-left quarter of the first YX plane"""
    reader = bioio_ome_zarr.Reader(test_file)
    reader.get_image_data(
        "YX", Y=slice(0, reader.dims.Y // 2), X=slice(0, reader.dims.X // 2)
    )


benchmark_functions: bioio_base.benchmark.BenchmarkDefinition = [
    {
        "prefix": "Get resolution levels",
        "test": lambda test_file: bioio_ome_zarr.Reader(test_file).resolution_levels,
    },
    {
        "prefix": "Get image data sub-region",
        "test": read_sub_region,
    },
]


# This file is under /scripts while the test resourcess are under /bioio_ome_zarr/tests/resources
test_resources_dir = pathlib.Path(__file__).parent.parent / "bioio_ome_zarr" / "tests" / "resources"
test_files = [
    test_file
    for test_file in test_resources_dir.iterdir()
    if test_file.name.endswith(".zarr")
]
print(f"Test files: {[file.name for file in test_files]}")
bioio_base.benchmark.benchmark(bioio_ome_zarr.reader.Reader, test_files, benchmark_functions)
