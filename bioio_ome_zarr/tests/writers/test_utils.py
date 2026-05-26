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
    pyramid_levels_to_tile_target,
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
    "old_pps, new_pps",
    [
        ([2.0, 0.5, 0.5], [4.0, 1.0, 1.0]),
    ],
)
def test_v3_edit_physical_pixel_size_propagates(
    tmp_path: pathlib.Path,
    old_pps: List[float],
    new_pps: List[float],
) -> None:
    store = str(tmp_path / "v3_pps.zarr")
    data = np.zeros((2, 4, 4), dtype=np.uint8)

    w = OMEZarrWriter(
        store=store,
        level_shapes=[(2, 4, 4), (2, 2, 2), (2, 1, 1)],
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


# ---------------------------------------------------------------------------
# pyramid_levels_to_tile_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level0_shape, canvas_size, n_spatial, expected_levels",
    [
        # Z=1 — single tile (1×1 grid), same as a plain Y/X bound.
        (
            (1, 4096, 4096),
            2048,
            3,
            [(1, 4096, 4096), (1, 2048, 2048)],
        ),
        # Z=4 — 2×2 grid; each tile must fit in canvas/2 = 1024.
        (
            (4, 2048, 2048),
            2048,
            3,
            [(4, 2048, 2048), (4, 1024, 1024)],
        ),
        # Z=9 — 3×3 grid; tiles need two halvings to fit in canvas/3 ≈ 682.
        (
            (9, 2048, 2048),
            2048,
            3,
            [(9, 2048, 2048), (9, 1024, 1024), (9, 512, 512)],
        ),
        # Z=50 — 8×7 grid; requires many halvings to fit (each tile ~256).
        (
            (50, 8192, 8192),
            2048,
            3,
            [
                (50, 8192, 8192),
                (50, 4096, 4096),
                (50, 2048, 2048),
                (50, 1024, 1024),
                (50, 512, 512),
                (50, 256, 256),
            ],
        ),
        # n_spatial=2 — no Z axis; grid is always 1×1 (plain Y/X bound).
        (
            (50, 4096, 4096),
            2048,
            2,
            [(50, 4096, 4096), (50, 2048, 2048)],
        ),
        # 5D TCZYX — non-spatial T and C must never change.
        (
            (2, 3, 9, 2048, 2048),
            2048,
            3,
            [(2, 3, 9, 2048, 2048), (2, 3, 9, 1024, 1024), (2, 3, 9, 512, 512)],
        ),
        # Z=2048 — symmetric ZYX cube; Z co-halves every step (always == Y/X),
        # producing 5 levels down to (128, 128, 128) where an 11×12 grid fits.
        (
            (2048, 2048, 2048),
            2048,
            3,
            [
                (2048, 2048, 2048),
                (1024, 1024, 1024),
                (512, 512, 512),
                (256, 256, 256),
                (128, 128, 128),
            ],
        ),
        # Z=200 — Z stays fixed while Y/X halve (200 < Y at each step), then
        # at the Y=256→128 transition next_min_inner=128 < Z=200, so Z
        # co-halves from 200→100 in the same step as Y/X.
        (
            (200, 4096, 4096),
            2048,
            3,
            [
                (200, 4096, 4096),
                (200, 2048, 2048),
                (200, 1024, 1024),
                (200, 512, 512),
                (200, 256, 256),
                (100, 128, 128),
            ],
        ),
        # level0 already fits — single-element list returned immediately.
        (
            (1, 512, 512),
            2048,
            3,
            [(1, 512, 512)],
        ),
        # 5D with large Z — exercises non-spatial axis invariance across many levels.
        (
            (2, 3, 50, 16384, 16384),
            2048,
            3,
            [
                (2, 3, 50, 16384, 16384),
                (2, 3, 50, 8192, 8192),
                (2, 3, 50, 4096, 4096),
                (2, 3, 50, 2048, 2048),
                (2, 3, 50, 1024, 1024),
                (2, 3, 50, 512, 512),
                (2, 3, 50, 256, 256),
            ],
        ),
        # trivial all-ones — fits immediately, verifies no-duplicate guard.
        (
            (1, 1, 1),
            2048,
            3,
            [(1, 1, 1)],
        ),
    ],
)
def test_tile_target_shape_sequence(
    level0_shape: Tuple[int, ...],
    canvas_size: int,
    n_spatial: int,
    expected_levels: List[Tuple[int, ...]],
) -> None:
    import math as _math

    levels = pyramid_levels_to_tile_target(
        level0_shape, canvas_size=canvas_size, n_spatial=n_spatial
    )

    # Exact sequence matches expected prefix.
    assert levels[: len(expected_levels)] == [tuple(s) for s in expected_levels]

    # Level 0 is always the first element.
    assert levels[0] == tuple(level0_shape)

    # No consecutive duplicate levels.
    for a, b in zip(levels, levels[1:]):
        assert a != b, f"Duplicate consecutive level {a}"

    # Non-spatial axes are unchanged across all levels.
    ndim = len(level0_shape)
    n_sp = min(n_spatial, ndim)
    non_spatial_end = ndim - n_sp
    for lvl in levels:
        assert lvl[:non_spatial_end] == tuple(
            level0_shape[:non_spatial_end]
        ), f"Non-spatial axes changed: {lvl}"

    # Helper: check whether a shape's Z-plane grid fits within canvas_size.
    def _grid_fits(shape: Tuple[int, ...]) -> bool:
        if n_sp >= 3:
            z = _math.prod(shape[ndim - n_sp : ndim - 2])
            cols = _math.ceil(_math.sqrt(z)) if z > 0 else 1
            rows = _math.ceil(z / cols) if cols > 0 else 1
        else:
            rows, cols = 1, 1
        y, x = shape[-2], shape[-1]
        return rows * y <= canvas_size and cols * x <= canvas_size

    # Bottom level must fit.
    assert _grid_fits(
        levels[-1]
    ), f"Bottom level {levels[-1]} does not fit in {canvas_size}×{canvas_size} canvas"

    # Second-to-last level must NOT fit (pyramid is maximally coarse).
    if len(levels) >= 2:
        assert not _grid_fits(
            levels[-2]
        ), f"Second-to-last {levels[-2]} already fit — pyramid stopped too early"

    # Validate that expected_levels is mathematically consistent with the
    # canvas constraint: every level except the last must NOT fit, and the
    # last must fit.
    expected_tuples = [tuple(s) for s in expected_levels]
    for shape in expected_tuples[:-1]:
        assert not _grid_fits(shape), (
            f"Expected non-fitting level {shape} actually fits the "
            f"{canvas_size}×{canvas_size} canvas — expected_levels is wrong"
        )
    assert _grid_fits(expected_tuples[-1]), (
        f"Expected bottom level {expected_tuples[-1]} does not fit the "
        f"{canvas_size}×{canvas_size} canvas — expected_levels is wrong"
    )
