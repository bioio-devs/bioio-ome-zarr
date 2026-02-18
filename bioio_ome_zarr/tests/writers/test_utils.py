import pathlib

import numpy as np
from dask import array as da

from bioio_ome_zarr import Reader
from bioio_ome_zarr.writers import (
    Channel,
    OMEZarrWriter,
    add_zarr_level,
    resize,
)


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
