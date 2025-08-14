#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import shutil
import tempfile
from typing import Any, Callable, List, Tuple

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import from_ngff_zarr
from ngff_zarr.validate import validate

from bioio_ome_zarr.writers import (
    OmeZarrWriterV3,  # keep this alias to match existing suite
)
from bioio_ome_zarr.writers import (
    DimTuple,
    chunk_size_from_memory_target,
)

from ..conftest import array_constructor

# -----------------------
# Helpers
# -----------------------


def _scales_from_expected_shapes(
    expected_shapes: List[Tuple[int, ...]],
) -> Tuple[Tuple[float, ...], ...]:
    """
    Build per-level 'scale' entries (relative sizes vs. level-0) from
    the expected_shapes already declared in the parametrization.

    For L > 0:
      scale[L-1][i] = shape_L[i] / shape_0[i]
    """
    base = np.array(expected_shapes[0], dtype=float)
    scales: List[Tuple[float, ...]] = []
    for s in expected_shapes[1:]:
        scales.append(tuple(np.array(s, dtype=float) / base))
    return tuple(scales)


def _spatial_mask_from_axes_types(axes_types: List[str]) -> List[bool]:
    return [t == "space" for t in axes_types]


def _scales_from_factors_until_one(
    base_shape: Tuple[int, ...],
    factors: Tuple[int, ...],
    axes_types: List[str],
) -> Tuple[Tuple[float, ...], ...]:
    """
    Produce a sequence of 'scale' entries from integer per-axis factors,
    reducing only spatial axes by their factor each level until no change.

    Relative size vs level-0 for level k (k>=1):
      spatial: 1 / (factor ** k)
      non-spatial: 1.0
    """
    spatial_mask = _spatial_mask_from_axes_types(axes_types)
    ndim = len(base_shape)
    curr = list(base_shape)
    k = 1
    scales: List[Tuple[float, ...]] = []
    while True:
        rel: List[float] = []
        would_change = False
        for i in range(ndim):
            if spatial_mask[i]:
                rel_i = 1.0 / (float(factors[i]) ** float(k))
                next_abs = max(1, int(np.floor(base_shape[i] * rel_i)))
                if next_abs < curr[i]:
                    would_change = True
                rel.append(rel_i)
            else:
                rel.append(1.0)
        if not would_change:
            break
        scales.append(tuple(rel))
        curr = [
            max(1, int(np.floor(base_shape[i] * scales[-1][i]))) for i in range(ndim)
        ]
        k += 1
    return tuple(scales)


# -----------------------
# Existing chunk helper test (unchanged)
# -----------------------


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


# -----------------------
# Legacy V2 writer parity — now using new API
# -----------------------


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

    # v2 + NGFF 0.4 with NEW API: derive per-level 'scale' from expected shapes
    save_uri = tmp_path / filename
    scale = _scales_from_expected_shapes(expected_shapes)

    writer = OmeZarrWriterV3(
        store=str(save_uri),
        shape=shape,
        dtype=im.dtype,
        axes_names=["t", "c", "z", "y", "x"],
        zarr_format=2,  # write Zarr v2
        image_name="TEST",
        scale=scale,  # <- per-level relative sizes vs level 0
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

    # v2 + NGFF 0.4 with NEW API
    save_uri = tmp_path / filename
    scale = _scales_from_expected_shapes(expected_shapes)

    writer = OmeZarrWriterV3(
        store=str(save_uri),
        shape=shape,
        dtype=im.dtype,
        axes_names=["t", "c", "z", "y", "x"],
        zarr_format=2,
        image_name="TEST",
        scale=scale,
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


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, data_generator, expected_shapes",
    [
        # 2D (YX) — automatic axes defaults (last N of t,c,z,y,x)
        (
            (4, 4),
            None,
            None,
            lambda: np.arange(16, dtype=np.uint8).reshape((4, 4)),
            [(4, 4), (2, 2), (1, 1)],
        ),
        # 3D (TYX) — time + 2D spatial
        (
            (4, 8, 8),
            None,
            None,
            lambda: np.random.randint(0, 255, size=(4, 8, 8), dtype=np.uint8),
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
        ),
        # 4D (TZYX)
        (
            (3, 4, 8, 8),
            ["t", "z", "y", "x"],
            ["time", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(3, 4, 8, 8), dtype=np.uint8),
            [(3, 4, 8, 8), (3, 4, 4, 4), (3, 4, 2, 2), (3, 4, 1, 1)],
        ),
        # 4D (CZYX)
        (
            (2, 4, 8, 8),
            ["c", "z", "y", "x"],
            ["channel", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(2, 4, 8, 8), dtype=np.uint8),
            [(2, 4, 8, 8), (2, 4, 4, 4), (2, 4, 2, 2), (2, 4, 1, 1)],
        ),
        # 5D (TCZYX) minimal case
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(1, 1, 1, 4, 4), dtype=np.uint8),
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2), (1, 1, 1, 1, 1)],
        ),
    ],
)
def test_write_full_volume_and_metadata_v2_lowerdim(
    shape: Tuple[int, ...],
    axes_names: Any,
    axes_types: Any,
    data_generator: Any,
    expected_shapes: List[Tuple[int, ...]],
) -> None:
    tmpdir = tempfile.mkdtemp()
    try:
        # Arrange
        data = data_generator()
        scale = _scales_from_expected_shapes(expected_shapes)

        writer_kwargs = {
            "store": tmpdir,
            "shape": shape,
            "dtype": data.dtype,
            "scale": scale,  # per-level relative sizes vs level 0
            "zarr_format": 2,  # NGFF 0.4
        }
        if axes_names:
            writer_kwargs["axes_names"] = axes_names
        if axes_types:
            writer_kwargs["axes_types"] = axes_types

        writer = OmeZarrWriterV3(**writer_kwargs)

        # Act
        writer.write_full_volume(data)

        # Assert: arrays exist with expected shapes
        grp = zarr.open(tmpdir, mode="r")
        for idx, exp_shape in enumerate(expected_shapes):
            arr = grp[str(idx)]
            assert arr.shape == exp_shape

        # Assert: NGFF 0.4 metadata is present and validates
        attrs = grp.attrs.asdict()
        assert "multiscales" in attrs and "omero" in attrs
        validate(attrs, version="0.4", model="image", strict=False)

        # Basic multiscale sanity
        ms = attrs["multiscales"][0]
        assert len(ms["datasets"]) == len(expected_shapes)
    finally:
        shutil.rmtree(tmpdir)


# NEW: v2 version of the “full vs timepoint” equivalence test
@pytest.mark.parametrize(
    "shape, axes_names, axes_types, factors",
    [
        # 3D (TYX)
        (
            (2, 4, 4),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 2, 2),
        ),
        # 3D (TYX) with different factors
        (
            (3, 6, 6),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 3, 3),
        ),
        # 3D (TYX) larger spatial with stronger downsample
        (
            (2, 128, 128),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 4, 4),
        ),
        # 5D (TCZYX)
        (
            (2, 2, 4, 128, 128),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (1, 1, 2, 4, 4),
        ),
    ],
)
def test_full_vs_timepoint_equivalence_v2(
    tmp_path: Any,
    shape: Tuple[int, ...],
    axes_names: List[str],
    axes_types: List[str],
    factors: Tuple[int, ...],
) -> None:
    """
    For Zarr v2 (NGFF 0.4), writing the full volume vs. one timepoint at a time
    yields identical arrays and metadata. (Sharding is not applicable in v2.)
    """
    # Arrange
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    full_store = str(tmp_path / "full_v2.zarr")
    tp_store = str(tmp_path / "tp_v2.zarr")

    # Build per-level 'scale' from integer per-axis factors
    scale = _scales_from_factors_until_one(
        shape, tuple(int(x) for x in factors), axes_types
    )

    w_full = OmeZarrWriterV3(
        store=full_store,
        shape=shape,
        dtype=data.dtype,
        axes_names=axes_names,
        axes_types=axes_types,
        scale=scale,
        zarr_format=2,  # NGFF 0.4
    )
    w_tp = OmeZarrWriterV3(
        store=tp_store,
        shape=shape,
        dtype=data.dtype,
        axes_names=axes_names,
        axes_types=axes_types,
        scale=scale,
        zarr_format=2,  # NGFF 0.4
    )

    # Act
    w_full.write_full_volume(data)
    for t in range(shape[0]):
        slice_data = data[t]
        # chunking is internal/per-level for v2; no sharding
        slice_da = da.from_array(slice_data)
        w_tp.write_timepoint(t, slice_da)

    # Assert
    grp_full = zarr.open(full_store, mode="r")
    grp_tp = zarr.open(tp_store, mode="r")

    # 1) Data equality per level
    for lvl, _ in enumerate(w_full.level_shapes):
        np.testing.assert_array_equal(grp_full[str(lvl)][...], grp_tp[str(lvl)][...])

    # 2) Metadata presence and basic sanity
    attrs_full = grp_full.attrs.asdict()
    attrs_tp = grp_tp.attrs.asdict()
    assert "multiscales" in attrs_full and "omero" in attrs_full
    assert "multiscales" in attrs_tp and "omero" in attrs_tp

    # Optional: validate both (relaxed)
    validate(attrs_full, version="0.4", model="image", strict=False)
    validate(attrs_tp, version="0.4", model="image", strict=False)

    # 3) Multiscale dataset count matches
    assert len(attrs_full["multiscales"][0]["datasets"]) == len(w_full.level_shapes)
    assert len(attrs_tp["multiscales"][0]["datasets"]) == len(w_tp.level_shapes)
