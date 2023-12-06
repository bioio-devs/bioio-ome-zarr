from typing import List, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities

from bioio_ome_zarr import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "set_resolution_level, "
    "expected_resolution_levels, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        pytest.param(
            "example.png",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        # General Zarr
        (
            "s1_t1_c1_z1_Image_0.zarr",
            "s1_t1_c1_z1",
            ("s1_t1_c1_z1",),
            2,
            (0, 1, 2, 3),
            (1, 1, 1, 7548, 7549),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
            (1.0, 1058.3333333333333, 1058.3333333333333),
        ),
        # Complex General Zarr
        (
            "s1_t7_c4_z3_Image_0.zarr",
            "s1_t7_c4_z3_Image_0",
            ("s1_t7_c4_z3_Image_0",),
            3,
            (0, 1, 2, 3),
            (7, 4, 3, 1200, 1800),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["C:0", "C:1", "C:2", "C:3"],
            (1.0, 8.0, 8.0),
        ),
        # Test Resolution Constant
        (
            "resolution_constant_zyx.zarr",
            "resolution_constant_zyx",
            ("resolution_constant_zyx",),
            2,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (0.1, 0.4, 0.4),
        ),
        # Test TYX
        (
            "dimension_handling_tyx.zarr",
            "dimension_handling_tyx",
            ("dimension_handling_tyx",),
            1,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.Time
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (None, 2.0, 2.0),
        ),
        # Test ZYX
        (
            "dimension_handling_zyx.zarr",
            "dimension_handling_zyx",
            ("dimension_handling_zyx",),
            1,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (1.0, 2.0, 2.0),
        ),
        # Test TZYX
        (
            "dimension_handling_tzyx.zarr",
            "dimension_handling_tzyx",
            ("dimension_handling_tzyx",),
            0,
            (0, 1, 2, 3),
            (2, 2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.Time
                + dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (1.0, 1.0, 1.0),
        ),
        (
            "absent_metadata_dims_zyx.zarr",
            "absent_metadata_dims_zyx",
            ("absent_metadata_dims_zyx",),
            3,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (1.0, 8.0, 8.0),
        ),
    ],
)
def test_ome_zarr_reader(
    filename: str,
    set_scene: str,
    set_resolution_level: int,
    expected_scenes: Tuple[str, ...],
    expected_resolution_levels: Tuple[int, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        set_resolution_level=set_resolution_level,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_resolution_levels=expected_resolution_levels,
        expected_current_resolution_level=set_resolution_level,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=dict,
    )
