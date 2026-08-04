from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities, types
from ome_types import to_dict
from zarr.core.group import GroupMetadata

from bioio_ome_zarr import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, set_scene, expected_scenes, set_resolution_level, "
    "expected_resolution_levels, expected_shape, expected_dtype, "
    "expected_dims_order, expected_channel_names, "
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
        (
            "s1_t1_c1_z1_Image_0_V3.zarr",
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
        (
            "s1_t1_c1_z1_Image_0_V3.zarr",
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
        (
            "s1_t7_c4_z3_Image_0_V3.zarr",
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
        (
            "s1_t7_c4_z3_Image_0_V3.zarr",
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
        (
            "resolution_constant_zyx_V3.zarr",
            "resolution_constant_zyx",
            ("resolution_constant_zyx",),
            0,
            (0, 1, 2),
            (2, 4, 4),
            np.int64,
            dimensions.DimensionNames.SpatialZ
            + dimensions.DimensionNames.SpatialY
            + dimensions.DimensionNames.SpatialX,
            ["Channel:0"],
            (0.1, 0.1, 0.1),
        ),
        (
            "dimension_handling_tyx_V3.zarr",
            "dimension_handling_tyx",
            ("dimension_handling_tyx",),
            0,
            (0, 1, 2),
            (2, 4, 4),
            np.int64,
            dimensions.DimensionNames.Time
            + dimensions.DimensionNames.SpatialY
            + dimensions.DimensionNames.SpatialX,
            ["Channel:0"],
            (None, 1.0, 1.0),
        ),
        (
            "dimension_handling_zyx_V3.zarr",
            "dimension_handling_zyx",
            ("dimension_handling_zyx",),
            0,
            (0, 1, 2),
            (2, 4, 4),
            np.int64,
            dimensions.DimensionNames.SpatialZ
            + dimensions.DimensionNames.SpatialY
            + dimensions.DimensionNames.SpatialX,
            ["Channel:0"],
            (1.0, 1.0, 1.0),
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
def test_ome_zarr_reader_v3(
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
    uri = LOCAL_RESOURCES_DIR / filename
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
    "filename, expected_image_ids, expected_channel_ids, expected_hexes",
    [
        pytest.param(
            "s1_t7_c4_z3_Image_0_V3.zarr",
            ["Image:0"],
            [f"Channel:{i}" for i in range(4)],
            ["ff0000", "00ff00", "0000ff", "ffff00"],
            id="full-dims-multi-channel",
        ),
        pytest.param(
            "resolution_constant_zyx_V3.zarr",
            [],
            [],
            [],
            marks=pytest.mark.xfail(
                reason="Unsupported dtype 'int64', expecting ValueError",
                strict=False,
            ),
            id="zyx-int64-xfail",
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

    # Fail Case
    if expected_image_ids is None:
        with pytest.raises(ValueError):
            _ = reader.ome_metadata
        return

    # Act
    ome_first = reader.ome_metadata
    ome_dict = to_dict(ome_first)
    ch_meta = (
        reader.metadata.attributes.get("ome", {}).get("omero", {}).get("channels", [])
    )

    # Assert
    assert len(ome_dict["images"]) == len(expected_image_ids)

    for idx, img in enumerate(ome_dict["images"]):
        assert img["id"] == expected_image_ids[idx]
        assert img["name"] == reader.scenes[idx]

        pix = img["pixels"]
        assert pix["dimension_order"].value == "XYZCT"

        # Validate dimension sizes
        assert pix["size_x"] == getattr(reader.dims, "X", 1)
        assert pix["size_y"] == getattr(reader.dims, "Y", 1)
        assert pix["size_z"] == getattr(reader.dims, "Z", 1)
        assert pix["size_c"] == getattr(reader.dims, "C", 1)
        assert pix["size_t"] == getattr(reader.dims, "T", 1)

        # Validate pixel type
        assert pix["type"].value == str(reader.dtype)

        # Validate physical pixel sizes match reader.scale
        assert pix["physical_size_x"] == reader.scale.X
        assert pix["physical_size_y"] == reader.scale.Y
        assert pix["physical_size_z"] == reader.scale.Z

        # Validate channel properties
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

    # Assert reader state is restored
    assert reader.current_scene_index == 0


@pytest.mark.parametrize(
    "filename, expected_dims",
    [
        (
            "s1_t7_c4_z3_Image_0_V3.zarr",
            {
                "T": ("time", types.ureg.millisecond, 1.0),
                # Channel present → type "channel" and default dimensionless unit
                "C": ("channel", types.ureg.dimensionless, 1.0),
                "Z": ("space", types.ureg.micrometer, 1.0),
                "Y": ("space", types.ureg.micrometer, 1.0),
                "X": ("space", types.ureg.micrometer, 1.0),
            },
        ),
        (
            "dimension_handling_tyx_V3.zarr",
            {
                "T": ("time", types.ureg.millisecond, 1.0),
                # No C / Z axis in this store
                "C": (None, None, None),
                "Z": (None, None, None),
                "Y": ("space", types.ureg.micrometer, 1.0),
                "X": ("space", types.ureg.micrometer, 1.0),
            },
        ),
    ],
)
def test_dimension_properties_from_axes(
    filename: str,
    expected_dims: Dict[
        str, Tuple[Optional[str], Optional[types.Unit], Optional[float]]
    ],
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
    for dim, (expected_type, expected_unit, expected_scale) in expected_dims.items():
        prop = dim_to_prop[dim]
        scale_val = dim_to_scale_val[dim]

        # Type from NGFF axes (or None)
        assert prop.type == expected_type

        # Unit: either None or a specific unit from the shared registry
        if expected_unit is None:
            assert prop.unit is None
        else:
            assert prop.unit == expected_unit

        # Scale value should match Scale.<dim> (or None)
        if expected_scale is None:
            assert scale_val is None
        else:
            assert scale_val == pytest.approx(expected_scale)


def test_read_ome_metadata_channels_no_color() -> None:
    uri = LOCAL_RESOURCES_DIR / "test_ngff_channel_no_color.zarr"
    reader = Reader(uri)
    assert reader.ome_metadata.images[0].pixels.channels[0].name == "random"


# Stores converted by bioio-conversion with include_provenance=True. Each carries
# a root "bioio_conversion" attributes block pointing at a standard_metadata.json
# sidecar holding the source reader's StandardMetadata.
@pytest.mark.parametrize(
    "filename, expected",
    [
        pytest.param(
            "provenance_czi.ome.zarr",
            {
                "objective": "20x/0.8Air",
                "binning": "4x4",
                "imaged_by": "ruiany",
                "imaging_datetime": datetime(
                    2019, 6, 27, 18, 33, 40, 619371, tzinfo=timezone.utc
                ),
                "row": None,
                "column": None,
                "position_index": None,
                "stage_position_x": 12345.67,
                "stage_position_y": 2345.89,
                # Single timepoint: the source measured no timing.
                "timelapse_interval": None,
                "total_time_duration": None,
            },
            id="czi",
        ),
        pytest.param(
            "provenance_ome_tiff.ome.zarr",
            {
                "objective": "20x/0.8Air",
                "binning": "4x4",
                "imaged_by": "ruiany",
                "imaging_datetime": datetime(2019, 6, 27, 18, 39, 25, 807000),
                "row": None,
                "column": None,
                "position_index": None,
                "stage_position_x": None,
                "stage_position_y": None,
                "timelapse_interval": None,
                # Recorded by the source reader, in seconds.
                "total_time_duration": timedelta(seconds=5.245),
            },
            id="ome-tiff",
        ),
        pytest.param(
            "provenance_nd2_plate.ome.zarr",
            {
                "objective": "10x/0.3",
                "binning": "1x1",
                "imaged_by": None,
                "imaging_datetime": datetime(
                    2021,
                    9,
                    28,
                    6,
                    55,
                    1,
                    935004,
                    tzinfo=timezone(timedelta(hours=-7)),
                ),
                # Plate-derived well, from a provenance reader opened with plate=96.
                "row": "4",
                "column": "3",
                "position_index": None,
                "stage_position_x": None,
                "stage_position_y": None,
                # Real acquisition timing, not the store's nominal T scale of 1.0.
                "timelapse_interval": timedelta(seconds=18.49526),
                "total_time_duration": timedelta(seconds=73.981041),
            },
            id="nd2-plate",
        ),
    ],
)
def test_standard_metadata_from_provenance(
    filename: str, expected: Dict[str, Any]
) -> None:
    """Provenance-only fields are surfaced from the standard_metadata sidecar."""
    reader = Reader(LOCAL_RESOURCES_DIR / filename)
    metadata = reader.standard_metadata

    for field, value in expected.items():
        assert getattr(metadata, field) == value, field
        # The same value is exposed as a reader property.
        assert getattr(reader, field) == value, field


@pytest.mark.parametrize(
    "filename, expected_size_x, expected_size_y",
    [
        ("provenance_czi.ome.zarr", 475, 325),
        ("provenance_ome_tiff.ome.zarr", 475, 325),
        ("provenance_nd2_plate.ome.zarr", 32, 32),
    ],
)
def test_standard_metadata_natively_derived_fields_win(
    filename: str, expected_size_x: int, expected_size_y: int
) -> None:
    """Fields the store itself describes stay natively derived, not read from the
    sidecar."""
    metadata = Reader(LOCAL_RESOURCES_DIR / filename).standard_metadata
    assert metadata.image_size_x == expected_size_x
    assert metadata.image_size_y == expected_size_y


def test_standard_metadata_without_provenance() -> None:
    """Without a provenance block, provenance-only fields stay unset."""
    metadata = Reader(LOCAL_RESOURCES_DIR / "s1_t1_c1_z1_Image_0_V3.zarr")
    standard_metadata = metadata.standard_metadata
    assert standard_metadata.objective is None
    assert standard_metadata.row is None
    assert standard_metadata.column is None
    assert standard_metadata.binning is None
    assert standard_metadata.imaged_by is None
    assert standard_metadata.imaging_datetime is None
    assert standard_metadata.stage_position_x is None
    assert standard_metadata.stage_position_y is None
    # Natively-derived fields are still populated.
    assert standard_metadata.image_size_x is not None
