#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pathlib
from typing import Callable, List, Tuple, cast

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import from_ngff_zarr

from bioio_ome_zarr import Reader
from bioio_ome_zarr.writers import (
    DimTuple,
    OmeZarrWriterV2,
    chunk_size_from_memory_target,
    compute_level_chunk_sizes_zslice,
    compute_level_shapes,
)

from ..conftest import array_constructor


@pytest.mark.parametrize(
    "input_shape, dtype, memory_target, expected_chunk_shape",
    [
        ((1, 1, 1, 128, 128), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((1, 1, 1, 127, 127), np.uint16, 1024, (1, 1, 1, 15, 15)),
        ((1, 1, 1, 129, 129), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((7, 11, 128, 128, 128), np.uint16, 1024, (1, 1, 8, 8, 8)),
    ],
)
def test_chunk_size_from_memory_target(
    input_shape: DimTuple,
    dtype: np.dtype,
    memory_target: int,
    expected_chunk_shape: DimTuple,
) -> None:
    chunk_shape = chunk_size_from_memory_target(input_shape, dtype, memory_target)
    assert chunk_shape == expected_chunk_shape


@array_constructor
@pytest.mark.parametrize(
    "shape, num_levels, scaling, expected_shapes",
    [
        (
            (4, 2, 2, 64, 32),  # easy, powers of two
            3,
            (1, 1, 1, 2, 2),  # downscale xy by two
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
        ),
        (
            (4, 2, 2, 8, 6),
            1,  # no downscaling
            (1, 1, 1, 1, 1),
            [(4, 2, 2, 8, 6)],
        ),
        (
            (1, 1, 1, 13, 23),  # start with odd dimensions
            3,
            (1, 1, 1, 2, 2),
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11), (1, 1, 1, 3, 5)],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_write_ome_zarr(
    array_constructor: Callable,
    filename: str,
    shape: DimTuple,
    num_levels: int,
    scaling: Tuple[float, float, float, float, float],
    expected_shapes: List[DimTuple],
    tmp_path: pathlib.Path,
) -> None:
    # TCZYX order, downsampling x and y only
    im = array_constructor(shape, dtype=np.uint8)
    C = shape[1]

    shapes = compute_level_shapes(shape, scaling, num_levels)
    chunk_sizes = compute_level_chunk_sizes_zslice(shapes)
    shapes_5d: list[DimTuple] = [cast(DimTuple, s) for s in shapes]
    chunk_sizes_5d: list[DimTuple] = [cast(DimTuple, c) for c in chunk_sizes]

    # Create an OmeZarrWriter object
    writer = OmeZarrWriterV2()

    # Initialize the store. Use s3 url or local directory path!
    save_uri = tmp_path / filename
    writer.init_store(str(save_uri), shapes_5d, chunk_sizes_5d, im.dtype)

    # Write the image
    writer.write_t_batches_array(im, channels=[], tbatch=4)

    # TODO: get this from source image
    physical_scale = {
        "c": 1.0,  # default value for channel
        "t": 1.0,
        "z": 1.0,
        "y": 1.0,
        "x": 1.0,
    }
    physical_units = {
        "x": "micrometer",
        "y": "micrometer",
        "z": "micrometer",
        "t": "minute",
    }
    meta = writer.generate_metadata(
        image_name="TEST",
        channel_names=[f"c{i}" for i in range(C)],
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(C)],
    )
    writer.write_metadata(meta)

    # Read written result and check basics
    ms = from_ngff_zarr(str(save_uri), validate=False)
    assert len(ms.images) == num_levels

    for level, shape in zip(range(num_levels), expected_shapes):
        img = ms.images[level]
        assert img.data.shape == shape

    axis_names = [ax.name for ax in ms.metadata.axes]
    assert "".join(axis_names).upper() == "TCZYX"


@array_constructor
@pytest.mark.parametrize(
    "shape, num_levels, scaling, expected_shapes",
    [
        (
            (4, 2, 2, 64, 32),  # easy, powers of two
            3,
            (1, 1, 1, 2, 2),  # downscale xy by two
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
        ),
        (
            (4, 2, 2, 8, 6),
            1,  # no downscaling
            (1, 1, 1, 1, 1),
            [(4, 2, 2, 8, 6)],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_write_ome_zarr_iterative(
    array_constructor: Callable,
    filename: str,
    shape: DimTuple,
    num_levels: int,
    scaling: Tuple[float, float, float, float, float],
    expected_shapes: List[DimTuple],
    tmp_path: pathlib.Path,
) -> None:
    # TCZYX order, downsampling x and y only
    im = array_constructor(shape, dtype=np.uint8)
    C = shape[1]

    shapes = compute_level_shapes(shape, scaling, num_levels)
    chunk_sizes = compute_level_chunk_sizes_zslice(shapes)
    shapes_5d: list[DimTuple] = [cast(DimTuple, s) for s in shapes]
    chunk_sizes_5d: list[DimTuple] = [cast(DimTuple, c) for c in chunk_sizes]

    # Create an OmeZarrWriter object
    writer = OmeZarrWriterV2()

    # Initialize the store. Use s3 url or local directory path!
    save_uri = tmp_path / filename
    writer.init_store(str(save_uri), shapes_5d, chunk_sizes_5d, im.dtype)

    # Write the image iteratively as if we only have one timepoint at a time
    for t in range(shape[0]):
        t4d = im[t]
        t5d = np.expand_dims(t4d, axis=0)
        writer.write_t_batches_array(t5d, channels=[], tbatch=1, toffset=t)

    # TODO: get this from source image
    physical_scale = {
        "c": 1.0,  # default value for channel
        "t": 1.0,
        "z": 1.0,
        "y": 1.0,
        "x": 1.0,
    }
    physical_units = {
        "x": "micrometer",
        "y": "micrometer",
        "z": "micrometer",
        "t": "minute",
    }
    meta = writer.generate_metadata(
        image_name="TEST",
        channel_names=[f"c{i}" for i in range(C)],
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(C)],
    )
    writer.write_metadata(meta)

    # Read written result and check basics
    ms = from_ngff_zarr(str(save_uri), validate=False)
    assert len(ms.images) == num_levels

    for level, shape in zip(range(num_levels), expected_shapes):
        img = ms.images[level]
        assert img.data.shape == shape

    # also verify that level-0 data round-trips correctly
    for t in range(shape[0]):
        np.testing.assert_array_equal(ms.images[0].data[t], im[t])


def test_append_resolution_level(tmp_path: pathlib.Path) -> None:
    # -----------------
    # Arrange
    # -----------------
    zarr_path = str(tmp_path / "pyramid.zarr")
    dtype = np.uint16

    # Initial levels 0–2 (T, C, Z, Y, X)
    initial_shapes = [
        (1, 1, 1, 64, 64),
        (1, 1, 1, 32, 32),
        (1, 1, 1, 16, 16),
    ]
    initial_chunks = list(initial_shapes)

    # Dummy data for level 0
    data0 = np.arange(np.prod(initial_shapes[0]), dtype=dtype).reshape(
        initial_shapes[0]
    )
    da0 = da.from_array(data0, chunks=initial_chunks[0])

    # Metadata parameters
    physical_dims = {"x": 1.0, "y": 1.0, "z": 1.0, "t": 1.0, "c": 1.0}
    physical_units = {"x": "µm", "y": "µm", "z": "µm", "t": "s"}
    image_name = "test"
    channel_names = [f"c{i}" for i in range(initial_shapes[0][1])]
    channel_colors = [0xFF0000] * initial_shapes[0][1]

    # Create initial writer and write levels 0–2 + metadata
    w1 = OmeZarrWriterV2()
    w1.init_store(zarr_path, initial_shapes, initial_chunks, dtype, overwrite=True)
    w1.write_t_batches_array(da0)
    meta1 = w1.generate_metadata(
        image_name=image_name,
        channel_names=channel_names,
        physical_dims=physical_dims,
        physical_units=physical_units,
        channel_colors=channel_colors,
    )
    w1.write_metadata(meta1)

    reader1 = Reader(zarr_path)
    assert reader1.resolution_levels == (0, 1, 2)

    # -----------------
    # Act
    # -----------------
    # Append level 3 at half the resolution of level 2
    new_shape = (1, 1, 1, 8, 8)
    shapes = [initial_shapes[i] for i in reader1.resolution_levels] + [new_shape]
    chunks = [initial_chunks[i] for i in reader1.resolution_levels] + [new_shape]

    w2 = OmeZarrWriterV2()
    w2.init_store(zarr_path, shapes, chunks, dtype, overwrite=False)
    w2.write_t_batches_array(da0)
    meta2 = w2.generate_metadata(
        image_name=image_name,
        channel_names=channel_names,
        physical_dims=physical_dims,
        physical_units=physical_units,
        channel_colors=channel_colors,
    )
    w2.write_metadata(meta2)

    reader2 = Reader(zarr_path)

    # -----------------
    # Assert
    # -----------------
    # 1) resolution_levels includes the new level
    assert reader2.resolution_levels == (0, 1, 2, 3)

    # 2) resolution_level_dims maps correctly
    expected_dims = {
        0: initial_shapes[0],
        1: initial_shapes[1],
        2: initial_shapes[2],
        3: new_shape,
    }
    assert reader2.resolution_level_dims == expected_dims

    # 3) get_image_data returns correct arrays per level
    for lvl in reader2.resolution_levels:
        reader2.set_resolution_level(lvl)
        data = reader2.get_image_data()
        assert isinstance(data, np.ndarray)
        assert data.shape == expected_dims[lvl]
        if lvl == 0:
            np.testing.assert_array_equal(data, data0)
