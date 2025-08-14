import json
import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr.validate import validate

from bioio_ome_zarr.reader import Reader
from bioio_ome_zarr.writers import Channel, OmeZarrWriterV3
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
    "zarr_format, shape, axes_names, scale, expected_shapes, literal_level1",
    [
        # 5D TCZYX, two levels (downsample Y/X)
        (
            2,
            (4, 2, 2, 64, 32),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5), (1, 1, 1, 0.25, 0.25)),
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
            None,
        ),
        (
            3,
            (4, 2, 2, 64, 32),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5), (1, 1, 1, 0.25, 0.25)),
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
            None,
        ),
        # 5D TCZYX, single level (no multiscale when scale=None)
        (2, (4, 2, 2, 8, 6), ["t", "c", "z", "y", "x"], None, [(4, 2, 2, 8, 6)], None),
        (3, (4, 2, 2, 8, 6), ["t", "c", "z", "y", "x"], None, [(4, 2, 2, 8, 6)], None),
        # 5D TCZYX, one level with non-even shapes
        (
            2,
            (1, 1, 1, 13, 23),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5),),
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11)],
            None,
        ),
        (
            3,
            (1, 1, 1, 13, 23),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5),),
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11)],
            None,
        ),
        # 5D TCZYX tiny literal check (level-1 expected values)
        (
            2,
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5),),
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
            np.array([[5, 7], [13, 15]], dtype=np.uint16),
        ),
        (
            3,
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5),),
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
            np.array([[5, 7], [13, 15]], dtype=np.uint16),
        ),
        # 2D YX, two levels with literal check
        (
            2,
            (4, 4),
            None,
            ((0.5, 0.5), (0.25, 0.25)),
            [(4, 4), (2, 2), (1, 1)],
            np.array([[5, 7], [13, 15]], dtype=np.uint8),
        ),
        (
            3,
            (4, 4),
            None,
            ((0.5, 0.5), (0.25, 0.25)),
            [(4, 4), (2, 2), (1, 1)],
            np.array([[5, 7], [13, 15]], dtype=np.uint8),
        ),
        # 3D default (ZYX), three levels (downsample Y/X only)
        (
            2,
            (4, 8, 8),
            None,
            ((1, 0.5, 0.5), (1, 0.25, 0.25), (1, 0.125, 0.125)),
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
            None,
        ),
        (
            3,
            (4, 8, 8),
            None,
            ((1, 0.5, 0.5), (1, 0.25, 0.25), (1, 0.125, 0.125)),
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
            None,
        ),
        # 4D CZYX, one level
        (
            2,
            (2, 4, 8, 8),
            ["c", "z", "y", "x"],
            ((1, 0.5, 0.5, 0.5),),
            [(2, 4, 8, 8), (2, 2, 4, 4)],
            None,
        ),
        (
            3,
            (2, 4, 8, 8),
            ["c", "z", "y", "x"],
            ((1, 0.5, 0.5, 0.5),),
            [(2, 4, 8, 8), (2, 2, 4, 4)],
            None,
        ),
        # 4D TZYX, two levels
        (
            2,
            (3, 4, 8, 8),
            ["t", "z", "y", "x"],
            ((1, 0.5, 0.5, 0.5), (1, 0.25, 0.25, 0.25)),
            [(3, 4, 8, 8), (3, 2, 4, 4), (3, 1, 2, 2)],
            None,
        ),
        (
            3,
            (3, 4, 8, 8),
            ["t", "z", "y", "x"],
            ((1, 0.5, 0.5, 0.5), (1, 0.25, 0.25, 0.25)),
            [(3, 4, 8, 8), (3, 2, 4, 4), (3, 1, 2, 2)],
            None,
        ),
        # Mixed factors on T and spatial axes
        (
            2,
            (4, 1, 3, 8, 8),
            ["t", "c", "z", "y", "x"],
            ((0.5, 1, 0.5, 0.5, 0.5),),
            [(4, 1, 3, 8, 8), (2, 1, 1, 4, 4)],
            None,
        ),
        (
            3,
            (4, 1, 3, 8, 8),
            ["t", "c", "z", "y", "x"],
            ((0.5, 1, 0.5, 0.5, 0.5),),
            [(4, 1, 3, 8, 8), (2, 1, 1, 4, 4)],
            None,
        ),
    ],
)
def test_write_pyramid(
    tmp_path: pathlib.Path,
    zarr_format: int,
    shape: DimTuple,
    axes_names: Optional[List[str]],
    scale: Optional[Tuple[Tuple[float, ...], ...]],
    expected_shapes: List[DimTuple],
    literal_level1: Optional[np.ndarray],
) -> None:
    # Arrange
    save_uri = tmp_path / "e.zarr"
    tiny_threshold = (1 * 1 * 1 * 4 * 4) if len(shape) == 5 else (4 * 4)
    dtype = np.uint16 if int(np.prod(shape)) <= tiny_threshold else np.uint8
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    kwargs = dict(
        store=str(save_uri),
        shape=shape,
        dtype=data.dtype,
        zarr_format=zarr_format,
        scale=scale,
        image_name="TEST",
    )
    if axes_names:
        kwargs["axes_names"] = axes_names

    writer = OmeZarrWriterV3(**kwargs)

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(str(save_uri), mode="r")
    for level, exp in enumerate(expected_shapes):
        assert grp[str(level)].shape == exp

    if literal_level1 is not None and len(expected_shapes) >= 2:
        np.testing.assert_array_equal(np.squeeze(grp["1"][:]), literal_level1)

    attrs = grp.attrs.asdict()
    _validate_attrs_by_format(attrs, zarr_format)

    if zarr_format == 2:
        assert len(attrs["multiscales"][0]["datasets"]) == len(expected_shapes)
    else:
        assert len(attrs["ome"]["multiscales"][0]["datasets"]) == len(expected_shapes)


@pytest.mark.parametrize(
    "writer_zarr_format, writer_axes, writer_shape, src_axes, src_shape, scale, "
    "expect_level1_literal, expect_error",
    [
        # 2D TCYX, one level, literal check
        (
            2,
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ((1, 1, 0.5, 0.5),),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        (
            3,
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ((1, 1, 0.5, 0.5),),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        # Mismatch (source lacks C)
        (
            2,
            ["t", "c", "y", "x"],
            (2, 1, 8, 8),
            ["t", "y", "x"],
            (2, 8, 8),
            ((1, 1, 0.5, 0.5),),
            None,
            ValueError,
        ),
        (
            3,
            ["t", "c", "y", "x"],
            (2, 1, 8, 8),
            ["t", "y", "x"],
            (2, 8, 8),
            ((1, 1, 0.5, 0.5),),
            None,
            ValueError,
        ),
    ],
)
def test_write_timepoints_array(
    tmp_path: Any,
    writer_zarr_format: int,
    writer_axes: List[str],
    writer_shape: Tuple[int, ...],
    src_axes: List[str],
    src_shape: Tuple[int, ...],
    scale: Optional[Tuple[Tuple[float, ...], ...]],
    expect_level1_literal: Optional[np.ndarray],
    expect_error: Optional[type],
) -> None:
    # Arrange
    src = np.arange(np.prod(src_shape), dtype=np.uint16).reshape(src_shape)
    out_store = tmp_path / "out_array.zarr"
    writer = OmeZarrWriterV3(
        store=str(out_store),
        shape=writer_shape,
        dtype=src.dtype,
        zarr_format=cast(Literal[2, 3], writer_zarr_format),
        axes_names=writer_axes,
        scale=scale,
        image_name="TEST",
    )
    arr = da.from_array(src, chunks=tuple(max(1, s // 2) for s in src.shape))

    # Act & Assert
    def run() -> None:
        writer.write_timepoints(arr, tbatch=2)

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
    "writer_zarr_format, writer_axes, writer_shape, src_axes, src_shape, scale, "
    "expect_level1_literal, expect_error",
    [
        # 2D TCYX, one level, literal check
        (
            2,
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ((1, 1, 0.5, 0.5),),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        (
            3,
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ((1, 1, 0.5, 0.5),),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        # 5D TCZYX, one level
        (
            2,
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ((1, 1, 1, 0.5, 0.5),),
            None,
            None,
        ),
        (
            3,
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ((1, 1, 1, 0.5, 0.5),),
            None,
            None,
        ),
        # Mismatch (source lacks C)
        (
            2,
            ["t", "c", "y", "x"],
            (2, 1, 8, 8),
            ["t", "y", "x"],
            (2, 8, 8),
            ((1, 1, 0.5, 0.5),),
            None,
            ValueError,
        ),
        (
            3,
            ["t", "c", "y", "x"],
            (2, 1, 8, 8),
            ["t", "y", "x"],
            (2, 8, 8),
            ((1, 1, 0.5, 0.5),),
            None,
            ValueError,
        ),
    ],
)
def test_write_timepoints_reader(
    tmp_path: Any,
    writer_zarr_format: int,
    writer_axes: List[str],
    writer_shape: Tuple[int, ...],
    src_axes: List[str],
    src_shape: Tuple[int, ...],
    scale: Optional[Tuple[Tuple[float, ...], ...]],
    expect_level1_literal: Optional[np.ndarray],
    expect_error: Optional[type],
) -> None:
    # Arrange: build a v2 (NGFF 0.4) source for Reader
    src = np.arange(np.prod(src_shape), dtype=np.uint16).reshape(src_shape)
    in_store = tmp_path / "in_reader.zarr"
    in_root = zarr.open_group(str(in_store), mode="w")
    arr = in_root.create_array(
        name="0",
        shape=src.shape,
        dtype=src.dtype,
        chunks=tuple(max(1, s // 2) for s in src.shape),
    )
    arr[:] = src
    in_root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a} for a in src_axes],
            "datasets": [{"path": "0"}],
        }
    ]
    reader = Reader(str(in_store))

    out_store = tmp_path / "out_reader.zarr"
    writer = OmeZarrWriterV3(
        store=str(out_store),
        shape=writer_shape,
        dtype=src.dtype,
        zarr_format=cast(Literal[2, 3], writer_zarr_format),
        axes_names=writer_axes,
        scale=scale,
        image_name="TEST",
    )

    # Act & Assert
    def run() -> None:
        writer.write_timepoints(reader, tbatch=2)

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
    "shape, chunk_size, shard_factor",
    [
        # 2D YX, shard 2x2
        ((16, 16), (4, 4), (2, 2)),
        # 3D TYX, shard 1x2x2
        ((2, 16, 16), (1, 4, 4), (1, 2, 2)),
        # 4D CZYX, shard 1x1x2x2
        ((2, 2, 16, 16), (1, 1, 4, 4), (1, 1, 2, 2)),
        # 4D TZYX, shard 1x1x2x2
        ((3, 2, 16, 16), (1, 1, 4, 4), (1, 1, 2, 2)),
        # 5D TCZYX, shard 1x1x1x2x2
        ((2, 2, 2, 16, 16), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2)),
    ],
)
def test_v3_sharding_and_chunking(
    tmp_path: Any,
    shape: Tuple[int, ...],
    chunk_size: Tuple[int, ...],
    shard_factor: Tuple[int, ...],
) -> None:
    # Arrange
    data = np.zeros(shape, dtype=np.uint8)
    store = str(tmp_path / "test_highdim_v3.zarr")
    writer = OmeZarrWriterV3(
        store=store,
        shape=shape,
        dtype=data.dtype,
        zarr_format=3,
        chunk_shape=(chunk_size,),
        shard_factor=shard_factor,
    )

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(store, mode="r")
    arr = grp["0"]
    assert arr.shape == shape
    assert arr.chunks == chunk_size
    expected_shard = tuple(chunk_size[i] * shard_factor[i] for i in range(len(shape)))
    assert arr.shards == expected_shard


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, axes_units, physical_pixel_size, scale, "
    "channel_kwargs, chunk_size, shard_factor, filename",
    [
        # 5D TCZYX metadata reference (levels to 1×1)
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            [None, None, None, "micrometer", "micrometer"],
            [1.0, 1.0, 1.0, 0.5, 0.5],
            ((1, 1, 1, 0.5, 0.5), (1, 1, 1, 0.25, 0.25)),
            {
                "label": "Ch0",
                "color": "FF0000",
                "active": True,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {"min": 0, "max": 255, "start": 0, "end": 255},
            },
            (1, 1, 1, 4, 4),
            (1, 1, 1, 2, 2),
            "reference_zarr.json",
        ),
        # 3D TYX metadata reference (levels to 2×2 then 1×1)
        (
            (2, 4, 4),
            ["t", "y", "x"],
            ["time", "space", "space"],
            [None, "micrometer", "micrometer"],
            [1.0, 0.5, 0.5],
            ((1, 0.5, 0.5), (1, 0.25, 0.25)),
            {"label": "Ch0", "color": "FF0000"},
            (1, 4, 4),
            (1, 2, 2),
            "reference_zarr_tyx.json",
        ),
    ],
)
def test_v3_metadata_against_reference(
    tmp_path: Any,
    shape: Any,
    axes_names: Any,
    axes_types: Any,
    axes_units: Any,
    physical_pixel_size: Any,
    scale: Any,
    channel_kwargs: Any,
    chunk_size: Any,
    shard_factor: Any,
    filename: Any,
) -> None:
    # Arrange
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    ch0 = Channel(**channel_kwargs)
    store_dir = str(tmp_path / "ref_test_v3.zarr")
    chunk_shape = tuple(tuple(chunk_size) for _ in range(1 + len(scale)))

    writer = OmeZarrWriterV3(
        store=store_dir,
        shape=shape,
        dtype="uint8",
        zarr_format=3,
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
        scale=scale,
        chunk_shape=chunk_shape,
        shard_factor=shard_factor,
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
    "zarr_format, shape, axes_names, axes_types, scale, chunk_size, shard_factor",
    [
        # 3D TYX small: 4->2->1
        (
            2,
            (2, 4, 4),
            ["t", "y", "x"],
            ["time", "space", "space"],
            ((1, 0.5, 0.5), (1, 0.25, 0.25)),
            (1, 2, 2),
            (1, 1, 1),
        ),
        (
            3,
            (2, 4, 4),
            ["t", "y", "x"],
            ["time", "space", "space"],
            ((1, 0.5, 0.5), (1, 0.25, 0.25)),
            (1, 2, 2),
            (1, 1, 1),
        ),
        # 3D TYX rectangular: 6->2->1 with /3 per level
        (
            2,
            (3, 6, 6),
            ["t", "y", "x"],
            ["time", "space", "space"],
            ((1, 1 / 3, 1 / 3), (1, 1 / 6, 1 / 6)),
            (1, 2, 2),
            (1, 1, 1),
        ),
        (
            3,
            (3, 6, 6),
            ["t", "y", "x"],
            ["time", "space", "space"],
            ((1, 1 / 3, 1 / 3), (1, 1 / 6, 1 / 6)),
            (1, 2, 2),
            (1, 1, 1),
        ),
        # 3D TYX large spatial: 128 -> 32 -> 8 -> 2 -> 1 (four levels)
        (
            2,
            (2, 128, 128),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (
                (1, 0.25, 0.25),
                (1, 0.0625, 0.0625),
                (1, 0.015625, 0.015625),
                (1, 0.0078125, 0.0078125),
            ),
            (1, 32, 32),
            (1, 2, 2),
        ),
        (
            3,
            (2, 128, 128),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (
                (1, 0.25, 0.25),
                (1, 0.0625, 0.0625),
                (1, 0.015625, 0.015625),
                (1, 0.0078125, 0.0078125),
            ),
            (1, 32, 32),
            (1, 2, 2),
        ),
        # 5D TCZYX large: spatial (Y/X) /4 per level; Z /2 until 1 (4 levels total)
        (
            2,
            (2, 2, 4, 128, 128),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (
                (1, 1, 0.5, 0.25, 0.25),
                (1, 1, 0.25, 0.0625, 0.0625),
                (1, 1, 0.25, 0.015625, 0.015625),
                (1, 1, 0.25, 0.0078125, 0.0078125),
            ),
            (1, 1, 2, 32, 32),
            (1, 1, 1, 2, 2),
        ),
        (
            3,
            (2, 2, 4, 128, 128),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (
                (1, 1, 0.5, 0.25, 0.25),
                (1, 1, 0.25, 0.0625, 0.0625),
                (1, 1, 0.25, 0.015625, 0.015625),
                (1, 1, 0.25, 0.0078125, 0.0078125),
            ),
            (1, 1, 2, 32, 32),
            (1, 1, 1, 2, 2),
        ),
    ],
)
def test_full_vs_timepoints_equivalence(
    tmp_path: Any,
    zarr_format: int,
    shape: Tuple[int, ...],
    axes_names: List[str],
    axes_types: List[str],
    scale: Tuple[Tuple[float, ...], ...],
    chunk_size: Tuple[int, ...],
    shard_factor: Tuple[int, ...],
) -> None:
    # Arrange
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    full_store = str(tmp_path / "full.zarr")
    tp_store = str(tmp_path / "tp.zarr")
    chunk_shape = tuple(tuple(chunk_size) for _ in range(1 + len(scale)))

    w_full = OmeZarrWriterV3(
        store=full_store,
        shape=shape,
        dtype=data.dtype,
        zarr_format=cast(Literal[2, 3], zarr_format),
        axes_names=axes_names,
        axes_types=axes_types,
        scale=scale,
        chunk_shape=chunk_shape,
        shard_factor=shard_factor,
    )
    w_tp = OmeZarrWriterV3(
        store=tp_store,
        shape=shape,
        dtype=data.dtype,
        zarr_format=cast(Literal[2, 3], zarr_format),
        axes_names=axes_names,
        axes_types=axes_types,
        scale=scale,
        chunk_shape=chunk_shape,
        shard_factor=shard_factor,
    )

    # Act
    w_full.write_full_volume(data)
    dask_data = da.from_array(data, chunks=chunk_size)
    w_tp.write_timepoints(dask_data, tbatch=1)

    # Assert
    grp_full = zarr.open(full_store, mode="r")
    grp_tp = zarr.open(tp_store, mode="r")
    for lvl, _ in enumerate(w_full.level_shapes):
        arr_full = grp_full[str(lvl)]
        arr_tp = grp_tp[str(lvl)]

        np.testing.assert_array_equal(arr_full[...], arr_tp[...])  # data
        assert arr_full.chunks == chunk_size  # chunks
        assert arr_tp.chunks == chunk_size
        if zarr_format == 3:  # shards (v3)
            expected_shard = tuple(c * s for c, s in zip(chunk_size, shard_factor))
            assert arr_full.shards == expected_shard
            assert arr_tp.shards == expected_shard
