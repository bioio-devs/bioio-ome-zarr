import json
import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr.validate import validate

from bioio_ome_zarr import Reader
from bioio_ome_zarr.writers import Channel, OMEZarrWriter
from bioio_ome_zarr.writers.utils import DimTuple

from ..conftest import LOCAL_RESOURCES_DIR


# Validation Helper
def _validate_attrs_by_format(attrs: Dict[str, Any], zarr_format: int) -> None:
    if zarr_format == 2:
        validate(attrs, version="0.4", model="image", strict=False)
        assert "multiscales" in attrs and "omero" in attrs
    else:
        validate(attrs, version="0.5", model="image", strict=False)
        assert "ome" in attrs and "multiscales" in attrs["ome"]


@pytest.mark.parametrize(
    "zarr_format, level_shapes, axes_names, literal_level1",
    [
        # 5D TCZYX, two extra levels (downsample Y/X)
        (
            2,
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
            ["t", "c", "z", "y", "x"],
            None,
        ),
        (
            3,
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
            ["t", "c", "z", "y", "x"],
            None,
        ),
        # 5D TCZYX, single level (no multiscale)
        (2, (4, 2, 2, 8, 6), ["t", "c", "z", "y", "x"], None),
        (3, (4, 2, 2, 8, 6), ["t", "c", "z", "y", "x"], None),
        # 5D TCZYX, two levels
        (
            2,
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11)],
            ["t", "c", "z", "y", "x"],
            None,
        ),
        (
            3,
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11)],
            ["t", "c", "z", "y", "x"],
            None,
        ),
        # 5D TCZYX literal check
        (
            2,
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
            ["t", "c", "z", "y", "x"],
            np.array([[5, 7], [13, 15]], dtype=np.uint16),
        ),
        (
            3,
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
            ["t", "c", "z", "y", "x"],
            np.array([[5, 7], [13, 15]], dtype=np.uint16),
        ),
        # 2D YX, three levels with literal check
        (
            2,
            [(4, 4), (2, 2), (1, 1)],
            None,
            np.array([[5, 7], [13, 15]], dtype=np.uint8),
        ),
        (
            3,
            [(4, 4), (2, 2), (1, 1)],
            None,
            np.array([[5, 7], [13, 15]], dtype=np.uint8),
        ),
        # 3D default (ZYX), 4 levels (downsample Y/X only)
        (
            2,
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
            None,
            None,
        ),
        (
            3,
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
            None,
            None,
        ),
        # 4D CZYX, 2 levels
        (
            2,
            [(2, 4, 8, 8), (2, 2, 4, 4)],
            ["c", "z", "y", "x"],
            None,
        ),
        (
            3,
            [(2, 4, 8, 8), (2, 2, 4, 4)],
            ["c", "z", "y", "x"],
            None,
        ),
        # 4D TZYX, 3 levels
        (
            2,
            [(3, 4, 8, 8), (3, 2, 4, 4), (3, 1, 2, 2)],
            ["t", "z", "y", "x"],
            None,
        ),
        (
            3,
            [(3, 4, 8, 8), (3, 2, 4, 4), (3, 1, 2, 2)],
            ["t", "z", "y", "x"],
            None,
        ),
        # Mixed factors on T and spatial axes
        (
            2,
            [(4, 1, 3, 8, 8), (2, 1, 1, 4, 4)],
            ["t", "c", "z", "y", "x"],
            None,
        ),
        (
            3,
            [(4, 1, 3, 8, 8), (2, 1, 1, 4, 4)],
            ["t", "c", "z", "y", "x"],
            None,
        ),
    ],
)
def test_write_pyramid(
    tmp_path: pathlib.Path,
    zarr_format: int,
    level_shapes: List[DimTuple],
    axes_names: Optional[List[str]],
    literal_level1: Optional[np.ndarray],
) -> None:
    # Arrange
    save_uri = tmp_path / "e.zarr"
    shape0 = level_shapes[0]
    tiny_threshold = (1 * 1 * 1 * 4 * 4) if len(shape0) == 5 else (4 * 4)
    dtype = np.uint16 if int(np.prod(shape0)) <= tiny_threshold else np.uint8
    data = np.arange(np.prod(shape0), dtype=dtype).reshape(shape0)

    kwargs = dict(
        store=str(save_uri),
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=zarr_format,
        image_name="TEST",
    )
    if axes_names:
        kwargs["axes_names"] = axes_names

    writer = OMEZarrWriter(**kwargs)

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(str(save_uri), mode="r")
    for level, exp in enumerate(level_shapes):
        assert grp[str(level)].shape == exp

    if literal_level1 is not None and len(level_shapes) >= 2:
        np.testing.assert_array_equal(np.squeeze(grp["1"][:]), literal_level1)

    attrs = grp.attrs.asdict()
    _validate_attrs_by_format(attrs, zarr_format)

    if zarr_format == 2:
        assert len(attrs["multiscales"][0]["datasets"]) == len(level_shapes)
    else:
        assert len(attrs["ome"]["multiscales"][0]["datasets"]) == len(level_shapes)

    # validate bioio_ome_zarr reader
    reader = Reader(str(save_uri))
    assert reader is not None
    assert len(reader.shape) >= len(level_shapes[0])
    assert reader.shape == level_shapes[0]


@pytest.mark.parametrize(
    "writer_zarr_format, writer_axes, writer_level_shapes, src_axes, src_shape, "
    "expect_level1_literal, expect_error",
    [
        # 2D TCYX, 2 levels, literal check
        (
            2,
            ["t", "c", "y", "x"],
            [(1, 1, 4, 4), (1, 1, 2, 2)],
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        (
            3,
            ["t", "c", "y", "x"],
            [(1, 1, 4, 4), (1, 1, 2, 2)],
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        # Mismatch (source lacks C)
        (
            2,
            ["t", "c", "y", "x"],
            [(2, 1, 8, 8), (2, 1, 4, 4)],
            ["t", "y", "x"],
            (2, 8, 8),
            None,
            ValueError,
        ),
        (
            3,
            ["t", "c", "y", "x"],
            [(2, 1, 8, 8), (2, 1, 4, 4)],
            ["t", "y", "x"],
            (2, 8, 8),
            None,
            ValueError,
        ),
    ],
)
def test_write_timepoints_array(
    tmp_path: Any,
    writer_zarr_format: int,
    writer_axes: List[str],
    writer_level_shapes: List[Tuple[int, ...]],
    src_axes: List[str],
    src_shape: Tuple[int, ...],
    expect_level1_literal: Optional[np.ndarray],
    expect_error: Optional[type],
) -> None:
    # Arrange
    src = np.arange(np.prod(src_shape), dtype=np.uint16).reshape(src_shape)
    out_store = tmp_path / "out_array.zarr"
    writer = OMEZarrWriter(
        store=str(out_store),
        level_shapes=writer_level_shapes,
        dtype=src.dtype,
        zarr_format=cast(Literal[2, 3], writer_zarr_format),
        axes_names=writer_axes,
        image_name="TEST",
    )
    arr = da.from_array(src, chunks=tuple(max(1, s // 2) for s in src.shape))

    # Act & Assert
    def run() -> None:
        writer.write_timepoints(arr)

    if expect_error:
        with pytest.raises(expect_error):
            run()
        return

    run()
    root = zarr.open_group(str(out_store), mode="r")
    np.testing.assert_array_equal(root["0"][:], src)
    if expect_level1_literal is not None:
        np.testing.assert_array_equal(root["1"][:], expect_level1_literal)


@pytest.mark.parametrize(
    "level0_shape, chunk_size, shard_per_level",
    [
        # 2D YX — chunk (4,4) tiles 16x16 with shard (4,4)
        ((16, 16), (4, 4), [(4, 4)]),
        # 3D TYX — chunk (1,4,4); shards tile: (2,4,4) for (2,16,16)
        ((2, 16, 16), (1, 4, 4), [(2, 4, 4)]),
        # 4D CZYX — chunk (1,1,4,4); shards (2,2,4,4) for (2,2,16,16)
        ((2, 2, 16, 16), (1, 1, 4, 4), [(2, 2, 4, 4)]),
        # 4D TZYX — chunk (1,1,4,4); shards (3,2,4,4) for (3,2,16,16)
        ((3, 2, 16, 16), (1, 1, 4, 4), [(3, 2, 4, 4)]),
        # 5D TCZYX — chunk (1,1,1,4,4); shards (2,2,2,4,4) for (2,2,2,16,16)
        ((2, 2, 2, 16, 16), (1, 1, 1, 4, 4), [(2, 2, 2, 4, 4)]),
    ],
)
def test_v3_sharding_and_chunking(
    tmp_path: Any,
    level0_shape: Tuple[int, ...],
    chunk_size: Tuple[int, ...],
    shard_per_level: List[Tuple[int, ...]],
) -> None:
    # Arrange
    level_shapes = [tuple(level0_shape)]
    data = np.zeros(level0_shape, dtype=np.uint8)
    store = str(tmp_path / "test_highdim_v3.zarr")
    writer = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=3,
        chunk_shape=chunk_size,
        shard_shape=shard_per_level,
    )

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(store, mode="r")
    arr = grp["0"]
    assert arr.shape == level0_shape
    assert arr.chunks == chunk_size
    assert arr.shards == shard_per_level[0]


@pytest.mark.parametrize(
    "level_shapes, axes_names, axes_types, axes_units, physical_pixel_size, "
    "channel_kwargs, base_chunk_size, shard_shapes, filename",
    [
        # 5D TCZYX metadata reference
        (
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2), (1, 1, 1, 1, 1)],
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            [None, None, None, "micrometer", "micrometer"],
            [1.0, 1.0, 1.0, 0.5, 0.5],
            {
                "label": "Ch0",
                "color": "FF0000",
                "active": True,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {"min": 0, "max": 255, "start": 0, "end": 255},
            },
            (1, 1, 1, 1, 1),
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2), (1, 1, 1, 1, 1)],
            "reference_zarr.json",
        ),
        # 3D TYX metadata reference
        (
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            [None, "micrometer", "micrometer"],
            [1.0, 0.5, 0.5],
            {"label": "Ch0", "color": "FF0000"},
            (1, 1, 1),
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],
            "reference_zarr_tyx.json",
        ),
    ],
)
def test_v3_metadata_against_reference(
    tmp_path: Any,
    level_shapes: Any,
    axes_names: Any,
    axes_types: Any,
    axes_units: Any,
    physical_pixel_size: Any,
    channel_kwargs: Any,
    base_chunk_size: Any,
    shard_shapes: Any,
    filename: Any,
) -> None:
    # Arrange
    shape0 = tuple(level_shapes[0])
    data = np.arange(np.prod(shape0), dtype=np.uint8).reshape(shape0)
    ch0 = Channel(**channel_kwargs)
    store_dir = str(tmp_path / "ref_test_v3.zarr")
    chunk_shape = [tuple(base_chunk_size) for _ in range(len(level_shapes))]

    writer = OMEZarrWriter(
        store=store_dir,
        level_shapes=level_shapes,
        dtype="uint8",
        zarr_format=3,
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
        chunk_shape=chunk_shape,
        shard_shape=shard_shapes,
        channels=[ch0],
        creator_info={"name": "pytest", "version": "0.1"},
    )

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(store_dir, mode="r")
    generated = grp.attrs.asdict()
    uri = LOCAL_RESOURCES_DIR / filename
    with open(uri, "r") as f:
        reference = json.load(f)
    assert generated["ome"] == reference["attributes"]["ome"]


@pytest.mark.parametrize(
    "zarr_format, level_shapes, axes_names, axes_types, chunk_size, shard_shape",
    [
        # 3D TYX small
        (
            2,
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 1, 1),
            None,
        ),
        (
            3,
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 1, 1),
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],
        ),
        # 3D TYX
        (
            2,
            [(3, 6, 6), (3, 2, 2), (3, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 1, 1),
            None,
        ),
        (
            3,
            [(3, 6, 6), (3, 2, 2), (3, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 1, 1),
            [(3, 6, 6), (3, 2, 2), (3, 1, 1)],
        ),
        # 3D TYX large spatial
        (
            2,
            [(2, 128, 128), (2, 32, 32), (2, 8, 8), (2, 2, 2), (2, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 32, 32),
            None,
        ),
        (
            3,
            [(2, 128, 128), (2, 32, 32), (2, 8, 8), (2, 2, 2), (2, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 32, 32),
            [
                (2, 64, 64),
                (2, 64, 64),
                (2, 64, 64),
                (2, 64, 64),
                (2, 64, 64),
            ],
        ),
        # 5D TCZYX explicit levels
        (
            2,
            [
                (2, 2, 4, 128, 128),
                (2, 2, 2, 32, 32),
                (2, 2, 1, 8, 8),
                (2, 2, 1, 2, 2),
                (2, 2, 1, 1, 1),
            ],
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (1, 1, 2, 32, 32),
            None,
        ),
        (
            3,
            [
                (1, 1, 2, 128, 128),
                (1, 1, 2, 64, 64),
                (1, 1, 2, 32, 32),
                (1, 1, 2, 16, 16),
                (1, 1, 2, 8, 8),
            ],
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (1, 1, 2, 32, 32),
            [
                (1, 1, 2, 64, 64),
                (1, 1, 2, 64, 64),
                (1, 1, 2, 64, 64),
                (1, 1, 2, 64, 64),
                (1, 1, 2, 64, 64),
            ],
        ),
    ],
)
def test_full_vs_timepoints_equivalence(
    tmp_path: Any,
    zarr_format: int,
    level_shapes: List[Tuple[int, ...]],
    axes_names: List[str],
    axes_types: List[str],
    chunk_size: Tuple[int, ...],
    shard_shape: Optional[List[Tuple[int, ...]]],
) -> None:
    # Arrange
    shape0 = level_shapes[0]
    data = np.arange(np.prod(shape0), dtype=np.uint8).reshape(shape0)
    full_store = str(tmp_path / "full.zarr")
    tp_store = str(tmp_path / "tp.zarr")
    chunk_shape = [tuple(chunk_size) for _ in range(len(level_shapes))]

    w_full = OMEZarrWriter(
        store=full_store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=cast(Literal[2, 3], zarr_format),
        axes_names=axes_names,
        axes_types=axes_types,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )
    w_tp = OMEZarrWriter(
        store=tp_store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=cast(Literal[2, 3], zarr_format),
        axes_names=axes_names,
        axes_types=axes_types,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )

    # Act
    w_full.write_full_volume(data)
    dask_data = da.from_array(data, chunks=chunk_size)
    w_tp.write_timepoints(dask_data)

    # Assert
    grp_full = zarr.open(full_store, mode="r")
    grp_tp = zarr.open(tp_store, mode="r")
    for lvl, _ in enumerate(w_full.level_shapes):
        arr_full = grp_full[str(lvl)]
        arr_tp = grp_tp[str(lvl)]
        np.testing.assert_array_equal(arr_full[...], arr_tp[...])  # data
        assert arr_full.chunks == chunk_shape[lvl]
        assert arr_tp.chunks == chunk_shape[lvl]
        if zarr_format == 3:  # shards (v3)
            assert shard_shape is not None
            expected_shard = shard_shape[lvl]
            assert arr_full.shards == expected_shard
            assert arr_tp.shards == expected_shard


VALID_LEVELS = [(1, 8, 8), (1, 4, 4)]


@pytest.mark.parametrize(
    "zarr_format, level_shapes, chunk_shape, shard_shape, match",
    [
        # ---------------- Structural validation ----------------
        # Empty level_shapes should fail
        (3, [], None, None, r"level_shapes cannot be empty"),
        # Per-level ndim mismatch
        (3, [(1, 8, 8), (1, 4)], None, None, r"level_shapes\[1] length 2 != ndim 3"),
        # Empty chunk_shape is invalid when explicitly provided
        (3, VALID_LEVELS, [], None, r"chunk_shape cannot be empty"),
        # Chunk ndim mismatch
        (3, VALID_LEVELS, (4, 4), None, r"chunk_shape length 2 != ndim 3"),
        # Chunk per-level count mismatch
        (
            3,
            VALID_LEVELS,
            [(1, 4, 4), (1, 4, 4), (1, 4, 4)],
            None,
            r"chunk_shape must have 2 entries \(per level\), got 3",
        ),
        # Empty shard_shape is invalid when explicitly provided
        (3, VALID_LEVELS, None, [], r"shard_shape cannot be empty"),
        # Shard ndim mismatch
        (
            3,
            VALID_LEVELS,
            None,
            [(2, 2), (2, 2)],
            r"shard_shape\[0] length 2 != ndim 3",
        ),
        # Shard per-level count mismatch
        (
            3,
            VALID_LEVELS,
            None,
            [(1, 2, 2)],
            r"shard_shape must have 2 entries \(per level\), got 1",
        ),
        # ---------------- Chunk validation ----------------
        # Chunk dimension must be >= 1
        (
            3,
            VALID_LEVELS,
            [(1, 0, 4), (1, 2, 2)],
            None,
            r"chunk_shape\[0]\[1] must be >= 1",
        ),
        # Per-level count mismatch after normalization
        (
            3,
            [(1, 8, 8), (1, 4, 4), (1, 2, 2)],
            [(1, 4, 4), (1, 2, 2)],
            None,
            r"chunk_shape must have 3 entries",
        ),
        # ---------------- Shard validation ----------------
        # Sharding not supported for Zarr v2
        (
            2,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 2), (1, 2, 2)],
            r"shard_shape is not supported for Zarr v2",
        ),
        # Shard per-level count mismatch
        (
            3,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 2)],
            r"shard_shape must have 2 entries",
        ),
        # Shard ndim mismatch
        (
            3,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 2), (2, 2)],
            r"shard_shape\[1] length 2 != ndim 3",
        ),
        # Shard dimension must be >= 1
        (
            3,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 0), (1, 2, 2)],
            r"shard_shape\[0]\[2] must be >= 1",
        ),
        # Shard must be a multiple of the chunk size (level 0)
        (
            3,
            VALID_LEVELS,
            [(1, 4, 4), (1, 2, 2)],
            [(1, 6, 8), (1, 2, 2)],  # 6 % 4 != 0
            r"must be a multiple of chunk_dim 4",
        ),
        # Shard must be a multiple of the chunk size (level 1)
        (
            3,
            VALID_LEVELS,
            [(1, 4, 4), (1, 2, 2)],
            [(1, 8, 8), (1, 3, 2)],  # 3 % 2 != 0 on Y axis
            r"must be a multiple of chunk_dim 2",
        ),
    ],
)
def test_writer_validation_errors(
    zarr_format: int,
    level_shapes: list[tuple[int, ...]],
    chunk_shape: list[tuple[int, ...]] | tuple[int, ...] | None,
    shard_shape: list[tuple[int, ...]] | None,
    match: str,
) -> None:
    """Ensure invalid configurations raise the expected ValueError."""

    kwargs = dict(
        store="in-memory.zarr",  # no disk write called in init.
        dtype=np.uint8,
        axes_names=["t", "y", "x"],
    )
    kwargs["zarr_format"] = zarr_format

    def build() -> OMEZarrWriter:
        return OMEZarrWriter(
            level_shapes=level_shapes,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            **kwargs,
        )

    with pytest.raises(ValueError, match=match):
        build()
