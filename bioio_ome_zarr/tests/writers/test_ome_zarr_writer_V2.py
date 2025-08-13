#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, List, Tuple

import numpy as np
import pytest
from ngff_zarr import from_ngff_zarr

from bioio_ome_zarr.writers import (
    DimTuple,
    OmeZarrWriterV3,
    chunk_size_from_memory_target,
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
            (4, 2, 2, 64, 32),  # powers of two
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
            (1, 1, 1, 13, 23),  # odd dims
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

    # Create a V3 writer configured to emit Zarr v2 + NGFF 0.4
    save_uri = tmp_path / filename
    writer = OmeZarrWriterV3(
        store=str(save_uri),
        shape=shape,
        dtype=im.dtype,
        scale_factors=tuple(int(s) for s in scaling),  # ensure ints
        num_levels=num_levels,
        axes_names=["t", "c", "z", "y", "x"],
        zarr_format=2,  # << write Zarr v2
        # ngff_version="0.4"
        image_name="TEST",
    )

    # Write the whole volume (writer handles pyramid + metadata)
    writer.write_full_volume(im)

    # Read written result and check basics
    ms = from_ngff_zarr(str(save_uri), validate=False, version="0.4")
    assert len(ms.images) == num_levels

    for level, shape_expected in zip(range(num_levels), expected_shapes):
        img = ms.images[level]
        assert img.data.shape == shape_expected

    axis_names = [ax.name for ax in ms.metadata.axes]
    assert "".join(axis_names).upper() == "TCZYX"
    # Sanity: channel count preserved
    assert C == ms.images[0].data.shape[1]


@array_constructor
@pytest.mark.parametrize(
    "shape, num_levels, scaling, expected_shapes",
    [
        (
            (4, 2, 2, 64, 32),
            3,
            (1, 1, 1, 2, 2),
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
        ),
        (
            (4, 2, 2, 8, 6),
            1,
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
    # TCZYX order
    im = array_constructor(shape, dtype=np.uint8)

    # V3 writer configured for Zarr v2 + NGFF 0.4
    save_uri = tmp_path / filename
    writer = OmeZarrWriterV3(
        store=str(save_uri),
        shape=shape,
        dtype=im.dtype,
        scale_factors=tuple(int(s) for s in scaling),
        num_levels=num_levels,
        axes_names=["t", "c", "z", "y", "x"],
        zarr_format=2,
        image_name="TEST",
    )

    # Write iteratively, one timepoint at a time
    for t in range(shape[0]):
        t4d = im[t]  # CZYX
        writer.write_timepoint(t, t4d)  # writer expands to TCZYX internally

    # Read written result and check basics
    ms = from_ngff_zarr(str(save_uri), validate=False, version="0.4")
    assert len(ms.images) == num_levels

    for level, shape_expected in zip(range(num_levels), expected_shapes):
        img = ms.images[level]
        assert img.data.shape == shape_expected

    # verify level-0 data round-trips correctly
    for t in range(shape[0]):
        np.testing.assert_array_equal(ms.images[0].data[t], im[t])
