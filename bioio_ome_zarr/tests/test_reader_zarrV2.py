from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities
from ome_types import to_dict
from zarr.core.group import GroupMetadata

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
            0,
            (0, 1, 2, 3),
            (1, 1, 1, 7548, 7549),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
            (1.0, 264.5833333333333, 264.5833333333333),
        ),
        # General Zarr Lower Resolution
        (
            "s1_t1_c1_z1_Image_0.zarr",
            "s1_t1_c1_z1",
            ("s1_t1_c1_z1",),
            1,
            (0, 1, 2, 3),
            (1, 1, 1, 3774, 3774),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0:0"],
            (1.0, 529.1666666666666, 529.1666666666666),
        ),
        # Complex General Zarr
        (
            "s1_t7_c4_z3_Image_0.zarr",
            "s1_t7_c4_z3_Image_0",
            ("s1_t7_c4_z3_Image_0",),
            0,
            (0, 1, 2, 3),
            (7, 4, 3, 1200, 1800),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["C:0", "C:1", "C:2", "C:3"],
            (1.0, 1.0, 1.0),
        ),
        # Complex General Zarr Lower Resolution
        (
            "s1_t7_c4_z3_Image_0.zarr",
            "s1_t7_c4_z3_Image_0",
            ("s1_t7_c4_z3_Image_0",),
            1,
            (0, 1, 2, 3),
            (7, 4, 3, 600, 900),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["C:0", "C:1", "C:2", "C:3"],
            (1.0, 2.0, 2.0),
        ),
        # Test Resolution Constant
        (
            "resolution_constant_zyx.zarr",
            "resolution_constant_zyx",
            ("resolution_constant_zyx",),
            0,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (0.1, 0.1, 0.1),
        ),
        # Test TYX
        (
            "dimension_handling_tyx.zarr",
            "dimension_handling_tyx",
            ("dimension_handling_tyx",),
            0,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.Time
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (None, 1.0, 1.0),
        ),
        # Test ZYX
        (
            "dimension_handling_zyx.zarr",
            "dimension_handling_zyx",
            ("dimension_handling_zyx",),
            0,
            (0, 1, 2, 3),
            (2, 4, 4),
            np.int64,
            (
                dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (1.0, 1.0, 1.0),
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
            (2, 0, 0),
            np.int64,
            (
                dimensions.DimensionNames.SpatialZ
                + dimensions.DimensionNames.SpatialY
                + dimensions.DimensionNames.SpatialX
            ),
            ["Channel:0"],
            (1.0, 8.0, 8.0),
        ),
        pytest.param(
            "bioformats_v2",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(
                raises=exceptions.UnsupportedFileFormatError,
                reason="(bioformats2raw.layout) not supported",
            ),
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
        expected_metadata_type=GroupMetadata,
        reader_kwargs={},
    )


@pytest.mark.parametrize(
    ["uri", "expected_resolution_level_dims"],
    [
        [
            (
                "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/"
                "hipsc_fov_nuclei_timelapse_dataset/"
                "hipsc_fov_nuclei_timelapse_data_used_for_analysis/"
                "baseline_colonies_fov_timelapse_dataset/20200323_09_small/"
                "raw.ome.zarr/"
            ),
            {
                0: (570, 2, 42, 1248, 1824),
                1: (570, 2, 42, 624, 912),
                2: (570, 2, 42, 312, 456),
                3: (570, 2, 42, 156, 228),
                4: (570, 2, 42, 78, 114),
            },
        ],
    ],
)
def test_resolution_level_dims(
    uri: str, expected_resolution_level_dims: Dict[int, Tuple[int]]
) -> None:
    # Act
    image_container = Reader(uri)
    # Assert
    assert image_container.resolution_level_dims == expected_resolution_level_dims


@pytest.mark.parametrize(
    "filename, expected_image_ids, expected_channel_ids, expected_hexes",
    [
        pytest.param(
            "s1_t7_c4_z3_Image_0.zarr",
            ["Image:0"],
            [f"Channel:{i}" for i in range(4)],
            ["ff0000", "00ff00", "0000ff", "ffff00"],
            id="legacy-full-dims",
        ),
        pytest.param(
            "resolution_constant_zyx.zarr",
            [],
            [],
            [],
            marks=pytest.mark.xfail(
                reason="Unsupported dtype 'int64', expecting ValueError",
                strict=False,
            ),
            id="legacy-zyx-dtype-xfail",
        ),
    ],
)
def test_ome_metadata(
    filename: str,
    expected_image_ids: List[str],
    expected_channel_ids: List[str],
    expected_hexes: List[str],
) -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri)

    # Act & Assert for xfail case
    if not expected_image_ids:
        with pytest.raises(ValueError):
            _ = reader.ome_metadata
        return

    # Act
    ome_obj = reader.ome_metadata
    ome_dict = to_dict(ome_obj)

    # Extract channel metadata directly from top-level 'omero'
    omero = reader.metadata.attributes.get("omero", {})
    ch_meta = omero.get("channels", [])

    # Assert
    assert len(ome_dict["images"]) == len(expected_image_ids)

    for idx, img in enumerate(ome_dict["images"]):
        assert img["id"] == expected_image_ids[idx]
        assert img["name"] == reader.scenes[idx]

        pix = img["pixels"]
        assert pix["dimension_order"].value == "XYZCT"

        # Dimensions
        assert pix["size_x"] == getattr(reader.dims, "X", 1)
        assert pix["size_y"] == getattr(reader.dims, "Y", 1)
        assert pix["size_z"] == getattr(reader.dims, "Z", 1)
        assert pix["size_c"] == getattr(reader.dims, "C", 1)
        assert pix["size_t"] == getattr(reader.dims, "T", 1)

        # Pixel type
        assert pix["type"].value == str(reader.dtype)

        # Physical sizes
        assert pix["physical_size_x"] == reader.scale.X
        assert pix["physical_size_y"] == reader.scale.Y
        assert pix["physical_size_z"] == reader.scale.Z

        # Channel metadata validation
        for ch_idx, ch in enumerate(pix["channels"]):
            src = ch_meta[ch_idx]
            assert ch["id"] == expected_channel_ids[ch_idx]
            assert ch["name"] == src.get("label", "")
            assert ch["color"]._original.lower() == expected_hexes[ch_idx]

            contrast = ch.get("contrast_method") or []
            if not src.get("active", True):
                assert "Off" in contrast
            if src.get("inverted", False):
                assert "inverted" in contrast

    # Ensure the reader resets scene index
    assert reader.current_scene_index == 0


@pytest.mark.parametrize(
    "filename, expected_dims",
    [
        (
            "s1_t7_c4_z3_Image_0.zarr",
            {
                "T": ("time", "millisecond", 1.0),
                "C": ("channel", None, 1.0),
                "Z": ("space", "micrometer", 1.0),
                "Y": ("space", "micrometer", 1.0),
                "X": ("space", "micrometer", 1.0),
            },
        ),
        (
            "dimension_handling_tyx.zarr",
            {
                "T": ("time", "millisecond", 1.0),
                "C": (None, None, None),
                "Z": (None, None, None),
                "Y": ("space", "micrometer", 1.0),
                "X": ("space", "micrometer", 1.0),
            },
        ),
    ],
)
def test_dimension_properties_from_axes(
    filename: str,
    expected_dims: Dict[str, Tuple[Optional[str], Optional[str], Optional[float]]],
) -> None:

    # Arrange
    uri = LOCAL_RESOURCES_DIR / filename
    r = Reader(uri)

    # Act
    dp = r.dimension_properties
    s = r.scale

    dim_to_prop = {
        "T": dp.T,
        "C": dp.C,
        "Z": dp.Z,
        "Y": dp.Y,
        "X": dp.X,
    }
    dim_to_scale_val = {
        "T": s.T,
        "C": s.C,
        "Z": s.Z,
        "Y": s.Y,
        "X": s.X,
    }

    # Assert
    for dim, (expected_type, expected_unit, expected_value) in expected_dims.items():

        prop = dim_to_prop[dim]
        scale_val = dim_to_scale_val[dim]

        # Type and unit should match what we expect from NGFF axes (or None)
        assert prop.type == expected_type
        assert prop.unit == expected_unit

        # Value should match scale.<dim> (or None)
        if expected_value is None:
            assert prop.value is None
            assert scale_val is None
        else:
            assert prop.value == pytest.approx(expected_value)
            assert scale_val == pytest.approx(expected_value)
