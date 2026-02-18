import pathlib
from typing import Optional, Tuple, Union

import numpy as np
import pytest
from dask import array as da

from bioio_ome_zarr import Reader
from bioio_ome_zarr.writers import (
    Channel,
    OMEZarrWriter,
    add_zarr_level,
    chunk_size_from_memory_target,
    resize,
)


@pytest.mark.parametrize(
    "shape, dtype, target, expected",
    [
        # original 5D uint16 cases
        ((1, 1, 1, 128, 128), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((1, 1, 1, 127, 127), np.uint16, 1024, (1, 1, 1, 15, 15)),
        ((1, 1, 1, 129, 129), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((7, 11, 128, 128, 128), np.uint16, 1024, (1, 1, 8, 8, 8)),
        # 2D uint8 (YX) with 1 KiB target
        ((256, 256), np.uint8, 1024, (32, 32)),
        # 3D uint8 (ZYX) with ~1 KiB target
        ((10, 20, 30), np.uint8, 1000, (5, 10, 15)),
        # 4D uint8 (CZYX) with 4 KiB target
        ((2, 4, 64, 64), np.uint8, 4096, (1, 2, 32, 32)),
        # 5D float32 (TCZYX) with 256 KiB target
        ((1, 1, 64, 64, 64), np.float32, 256 * 1024, (1, 1, 32, 32, 32)),
        # >5D without explicit order should xfail
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            np.uint8,
            1024,
            None,
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="Shapes >5D without `order` must raise",
            ),
        ),
    ],
)
def test_chunk_size_from_memory_target(
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    target: int,
    expected: Optional[Tuple[int, ...]],
) -> None:
    """
    Parameterized test for chunk_size_from_memory_target:
      - Valid 2D–5D cases (various dtypes & sizes)
      - >5D case xfails with ValueError when order=None
    """
    out = chunk_size_from_memory_target(shape, dtype, target)
    assert out == expected


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
