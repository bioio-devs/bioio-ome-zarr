from typing import List, Tuple

import numpy as np
import pytest
from bioio_base import dimensions

from bioio_ome_zarr import Reader

ome_host = "https://uk1s3.embassy.ebi.ac.uk/ebi-ngff-challenge-2024/"


@pytest.mark.parametrize(
    "uri, "
    "set_scene, "
    "expected_scenes, "
    "set_resolution_level, "
    "expected_resolution_levels, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes, "
    "expected_time_interval, "
    "expected_scale",
    [
        # General Zarr
        (
            f"{ome_host}fb416517-e36d-4bdb-98f9-47b644138d47.zarr",
            "Image:0",
            "Image:0",
            0,
            (0, 1, 2),
            (1, 1, 190, 617, 617),
            np.float32,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel 0"],
            (1.0, 1.0, 1.0),
            1.0,
            (1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        (
            f"{ome_host}4ffaeed2-fa70-4907-820f-8a96ef683095.zarr",
            "Image:0",
            "Image:0",
            0,
            (0, 1),
            (1, 2, 1, 512, 512),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["AF555-T1", "AF488-T2"],
            (1.0, 0.22007964065255714, 0.22007964065255714),
            1.0,
            (1.0, 1.0, 1.0, 0.22007964065255714, 0.22007964065255714),
        ),
        (
            f"{ome_host}c0e5d621-62cc-43a6-9dad-2ddab8959d17.zarr",
            "Image:0",
            "Image:0",
            0,
            (0, 1, 2),
            (163, 2, 1, 1024, 1024),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["GFP", "DsRed"],
            (1.0, 6.8746696041186794, 6.8746696041186794),
            1.8112843,
            (1.8112843, 1.0, 1.0, 6.8746696041186794, 6.8746696041186794),
        ),
    ],
)
def test_https_read_zarrV3(
    uri: str,
    set_scene: str,
    set_resolution_level: int,
    expected_scenes: Tuple[str, ...],
    expected_resolution_levels: Tuple[int, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
    expected_time_interval: float,
    expected_scale: Tuple[float, float, float, float, float],
) -> None:
    # Run checks
    image_container = Reader(uri)
    image_container.set_scene(set_scene)
    image_container.set_resolution_level(set_resolution_level)

    # ASSERT
    assert image_container.scenes == (expected_scenes,)
    assert image_container.current_scene == expected_scenes
    assert image_container.resolution_levels == expected_resolution_levels
    assert image_container.shape == expected_shape
    assert image_container.dtype == expected_dtype
    assert image_container.dims.order == expected_dims_order
    assert image_container.channel_names == expected_channel_names
    assert image_container.current_resolution_level == set_resolution_level

    # temporal and spatial scaling information
    assert image_container.physical_pixel_sizes == expected_physical_pixel_sizes
    assert image_container.time_interval == expected_time_interval
    assert image_container.scale == expected_scale
