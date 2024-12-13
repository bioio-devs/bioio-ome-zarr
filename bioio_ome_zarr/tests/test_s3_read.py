import numpy as np
import pytest

from bioio_ome_zarr import Reader


@pytest.mark.parametrize(
    ["prefix", "fs_kwargs"],
    [
        ["s3://allencell/aics/", dict(anon=True)],
        ["https://allencell.s3.amazonaws.com/aics/", dict()],
    ],
)
def test_ome_zarr_reader(prefix: str, fs_kwargs: dict) -> None:
    # ARRANGE
    uri = (
        prefix + "nuc-morph-dataset"
        "/hipsc_fov_nuclei_timelapse_dataset"
        "/hipsc_fov_nuclei_timelapse_data_used_for_analysis"
        "/baseline_colonies_fov_timelapse_dataset/20200323_09_small/raw.ome.zarr"
    )
    scene = "/"
    resolution_level = 0

    # ACT
    image_container = Reader(uri, fs_kwargs=fs_kwargs)
    image_container.set_scene(scene)
    image_container.set_resolution_level(resolution_level)

    # ASSERT
    assert image_container.scenes == (scene,)
    assert image_container.current_scene == scene
    assert image_container.resolution_levels == (0, 1, 2, 3, 4)
    assert image_container.shape == (570, 2, 42, 1248, 1824)
    assert image_container.dtype == np.uint16
    assert image_container.dims.order == "TCZYX"
    assert image_container.dims.shape == (570, 2, 42, 1248, 1824)
    assert image_container.channel_names == ["EGFP", "Bright"]
    assert image_container.current_resolution_level == resolution_level

    # pixel sized in (Z, Y, X) order
    assert image_container.physical_pixel_sizes == (
        0.7579,
        0.2708333333333333,
        0.2708333333333333,
    )
