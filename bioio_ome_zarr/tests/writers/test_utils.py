import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import zarr
from dask import array as da

from bioio_ome_zarr import Reader
from bioio_ome_zarr.writers import (
    Channel,
    OMEZarrWriter,
    add_zarr_level,
    resize,
)
from bioio_ome_zarr.writers.utils import edit_metadata


def test_resize_simple() -> None:
    """
    Test the resize utility for a small 2D case.
    """
    d = da.from_array(np.arange(16).reshape(4, 4), chunks=(2, 2))
    out = resize(d, (2, 2))
    assert out.shape == (2, 2)
    assert out.dtype == d.dtype


def test_add_zarr_level_using_reader(tmp_path: pathlib.Path) -> None:
    # ARRANGE:
    store_path = tmp_path / "test.zarr"

    data = np.arange(16, dtype="uint8").reshape((1, 1, 1, 4, 4))
    da_data = da.from_array(data, chunks=data.shape)

    # Initialize OME‑Zarr with a single resolution level
    writer = OMEZarrWriter(
        store=str(store_path),
        level_shapes=data.shape,
        dtype=data.dtype,
        zarr_format=2,
        compressor=None,
        image_name="test",
        channels=[Channel(label="A", color="#FFFFFF")],
        axes_names=["t", "c", "z", "y", "x"],
        axes_types=["time", "channel", "space", "space", "space"],
        axes_units=["unit", "unit", "unit", "unit", "unit"],
        physical_pixel_size=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    writer.write_full_volume(da_data)

    # ACT
    add_zarr_level(str(store_path), (1, 1, 1, 0.5, 0.5), compressor=None)

    # ASSERT
    rdr = Reader(str(store_path))

    # 1) Confirm the correct number of pyramid levels
    levels = list(rdr.resolution_levels)
    assert levels == [0, 1], f"Expected resolution levels [0,1], got {levels}"

    # 2) Confirm the dimensions for each level
    dims = rdr.resolution_level_dims
    assert dims[0] == (1, 1, 1, 4, 4), f"Level 0 dims mismatch: {dims[0]}"
    assert dims[1] == (1, 1, 1, 2, 2), f"Level 1 dims mismatch: {dims[1]}"

    # 3) Confirm that the downsampled data matches resize output
    data0 = rdr.get_image_dask_data("TCZYX", resolution_level=0).compute()
    data1 = rdr.get_image_dask_data("TCZYX", resolution_level=1).compute()
    expected = resize(da.from_array(data0), data1.shape, order=0).compute()

    assert np.array_equal(data1, expected), "Data mismatch after downsampling"


@pytest.mark.parametrize(
    "level_shapes, axes_names, axes_types, physical_pixel_size",
    [
        (
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],  # TYX pyramid
            ["t", "y", "x"],
            ["time", "space", "space"],
            [2.0, 0.5, 0.5],
        ),
    ],
)
def test_v2_edit_channel_label(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    axes_names: List[str],
    axes_types: List[str],
    physical_pixel_size: List[float],
) -> None:
    store = str(tmp_path / "v2_channel.zarr")
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    # ARRANGE: Create Image
    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=2,
        axes_names=axes_names,
        axes_types=axes_types,
        physical_pixel_size=physical_pixel_size,
        image_name="TEST",
        channels=[Channel(label="Ch0", color="FF0000")],
    )
    w.write_full_volume(data)

    # ACT: edit channel
    edit_metadata(
        store,
        channels=[Channel(label="NewLabel", color="00FF00")],
    )

    # ASSERT:
    rdr = Reader(store)
    assert rdr.channel_names == ["NewLabel"]


@pytest.mark.parametrize(
    "level_shapes, old_axes_names, new_axes_names, new_axes_types",
    [
        (
            [(2, 4, 4), (2, 2, 2)],
            ["t", "y", "x"],
            ["z", "y", "x"],
            ["space", "space", "space"],
        ),
    ],
)
def test_v2_edit_axes(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    old_axes_names: List[str],
    new_axes_names: List[str],
    new_axes_types: List[str],
) -> None:
    store = str(tmp_path / "v2_axes.zarr")
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    # ARRANGE: Create Image
    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=2,
        axes_names=old_axes_names,
        axes_types=["time", "space", "space"],
        image_name="TEST",
    )
    w.write_full_volume(data)

    # ACT: change axes
    edit_metadata(
        store,
        axes_names=new_axes_names,
        axes_types=new_axes_types,
    )

    # ASSERT:
    rdr = Reader(store)
    assert rdr._get_ome_dims() == tuple(n.upper() for n in new_axes_names)

    root = zarr.open_group(store, mode="r")
    attrs = root.attrs.asdict()
    axes = attrs["multiscales"][0]["axes"]
    assert [a["type"] for a in axes] == new_axes_types


@pytest.mark.parametrize(
    "level_shapes, old_pps, new_pps",
    [
        ([(2, 4, 4), (2, 2, 2), (2, 1, 1)], [2.0, 0.5, 0.5], [4.0, 1.0, 1.0]),
    ],
)
def test_v2_edit_physical_pixel_size(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    old_pps: List[float],
    new_pps: List[float],
) -> None:
    store = str(tmp_path / "v2_pps.zarr")
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    # ARRANGE: Create Image
    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=2,
        axes_names=["t", "y", "x"],
        axes_types=["time", "space", "space"],
        physical_pixel_size=old_pps,
        image_name="TEST",
    )
    w.write_full_volume(data)

    # ACTL: Update PPS
    edit_metadata(store, physical_pixel_size=new_pps)

    # ASSERT
    root = zarr.open_group(store, mode="r")
    ms0 = root.attrs.asdict()["multiscales"][0]
    ds = ms0["datasets"]

    scale0 = ds[0]["coordinateTransformations"][0]["scale"]
    scale1 = ds[1]["coordinateTransformations"][0]["scale"]
    scale2 = ds[2]["coordinateTransformations"][0]["scale"]

    assert scale0 == new_pps

    assert scale1[0] == pytest.approx(new_pps[0])
    assert scale1[1] == pytest.approx(new_pps[1] * 2.0)
    assert scale1[2] == pytest.approx(new_pps[2] * 2.0)

    assert scale2[0] == pytest.approx(new_pps[0])
    assert scale2[1] == pytest.approx(new_pps[1] * 4.0)
    assert scale2[2] == pytest.approx(new_pps[2] * 4.0)


@pytest.mark.parametrize(
    "level_shapes, axes_names, axes_types, physical_pixel_size",
    [
        (
            [(2, 4, 4), (2, 2, 2), (2, 1, 1)],
            ["t", "y", "x"],
            ["time", "space", "space"],
            [2.0, 0.5, 0.5],
        ),
    ],
)
def test_v3_edit_channel_label(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    axes_names: List[str],
    axes_types: List[str],
    physical_pixel_size: List[float],
) -> None:
    store = str(tmp_path / "v3_channel.zarr")
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=3,
        axes_names=axes_names,
        axes_types=axes_types,
        physical_pixel_size=physical_pixel_size,
        image_name="TEST",
        channels=[Channel(label="Ch0", color="FF0000")],
    )
    w.write_full_volume(data)

    # ACT
    edit_metadata(
        store,
        channels=[Channel(label="NewLabel", color="00FF00")],
    )

    # ASSERT: Reader + raw attrs
    rdr = Reader(store)
    assert rdr.channel_names == ["NewLabel"]

    root = zarr.open_group(store, mode="r")
    attrs: Dict[str, Any] = root.attrs.asdict()
    assert attrs["ome"]["omero"]["channels"][0]["label"] == "NewLabel"


@pytest.mark.parametrize(
    "level_shapes, old_axes_names, new_axes_names, new_axes_types",
    [
        (
            [(2, 4, 4), (2, 2, 2)],
            ["t", "y", "x"],
            ["z", "y", "x"],
            ["space", "space", "space"],
        ),
    ],
)
def test_v3_edit_axes_tyx_to_zyx(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    old_axes_names: List[str],
    new_axes_names: List[str],
    new_axes_types: List[str],
) -> None:
    store = str(tmp_path / "v3_axes.zarr")
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=3,
        axes_names=old_axes_names,
        axes_types=["time", "space", "space"],
        image_name="TEST",
    )
    w.write_full_volume(data)

    # ACT
    edit_metadata(
        store,
        axes_names=new_axes_names,
        axes_types=new_axes_types,
    )

    # ASSERT: Reader sees new dims
    rdr = Reader(store)
    assert rdr._get_ome_dims() == tuple(n.upper() for n in new_axes_names)

    root = zarr.open_group(store, mode="r")
    attrs = root.attrs.asdict()
    axes = attrs["ome"]["multiscales"][0]["axes"]
    assert [a["name"] for a in axes] == new_axes_names
    assert [a["type"] for a in axes] == new_axes_types


@pytest.mark.parametrize(
    "level_shapes, old_pps, new_pps",
    [
        ([(2, 4, 4), (2, 2, 2), (2, 1, 1)], [2.0, 0.5, 0.5], [4.0, 1.0, 1.0]),
    ],
)
def test_v3_edit_physical_pixel_size_propagates(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    old_pps: List[float],
    new_pps: List[float],
) -> None:
    store = str(tmp_path / "v3_pps.zarr")
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=3,
        axes_names=["t", "y", "x"],
        axes_types=["time", "space", "space"],
        physical_pixel_size=old_pps,
        image_name="TEST",
    )
    w.write_full_volume(data)

    # ACT
    edit_metadata(store, physical_pixel_size=new_pps)

    # ASSERT
    root = zarr.open_group(store, mode="r")
    ms0 = root.attrs.asdict()["ome"]["multiscales"][0]
    ds = ms0["datasets"]

    scale0 = ds[0]["coordinateTransformations"][0]["scale"]
    scale1 = ds[1]["coordinateTransformations"][0]["scale"]
    scale2 = ds[2]["coordinateTransformations"][0]["scale"]

    assert scale0 == new_pps
    assert scale1[0] == pytest.approx(new_pps[0])
    assert scale1[1] == pytest.approx(new_pps[1] * 2.0)
    assert scale1[2] == pytest.approx(new_pps[2] * 2.0)

    assert scale2[0] == pytest.approx(new_pps[0])
    assert scale2[1] == pytest.approx(new_pps[1] * 4.0)
    assert scale2[2] == pytest.approx(new_pps[2] * 4.0)


@pytest.mark.parametrize(
    "creator_info",
    [
        {"name": "pytest", "version": "9.9.9"},
        {"name": "scientist", "git_sha": "abc123"},
    ],
)
def test_v3_edit_creator_info(
    tmp_path: pathlib.Path,
    creator_info: Dict[str, Any],
) -> None:
    store = str(tmp_path / "v3_creator.zarr")
    level_shapes = [(2, 4, 4), (2, 2, 2)]
    data = np.zeros(level_shapes[0], dtype=np.uint8)

    w = OMEZarrWriter(
        store=store,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=3,
        axes_names=["t", "y", "x"],
        axes_types=["time", "space", "space"],
        image_name="TEST",
    )
    w.write_full_volume(data)

    # ACT
    edit_metadata(store, creator_info=creator_info)

    # ASSERT (raw metadata)
    root = zarr.open_group(store, mode="r")
    attrs = root.attrs.asdict()
    assert attrs["ome"]["_creator"] == creator_info
