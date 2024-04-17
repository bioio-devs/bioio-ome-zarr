import numpy as np

from bioio_ome_zarr import Reader


def test_ome_zarr_reader() -> None:
    # ARRANGE
    uri = (
        # Cannot use s3:// URL due to ome-zarr issue #369
        # "s3://allencell/aics/nuc_morph_data"
        "https://allencell.s3.amazonaws.com/aics/nuc_morph_data"
        "/data_for_analysis/baseline_colonies/20200323_09_small/raw.ome.zarr"
    )
    scene = "/"
    resolution_level = 0

    # ACT
    image_container = Reader(uri, fs_kwargs=dict(anon=True))
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
    assert image_container.channel_names == ["low_EGFP", "low_Bright"]
    assert image_container.current_resolution_level == resolution_level
    # pixel sized in (Z, Y, X) order
    assert image_container.physical_pixel_sizes == (
        0.53,
        0.2708333333333333,
        0.2708333333333333,
    )
