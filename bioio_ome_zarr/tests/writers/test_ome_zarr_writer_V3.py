import json
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import zarr
from ngff_zarr.validate import validate

from bioio_ome_zarr.writers import Channel, OmeZarrWriterV3, spatial_downsample

from ..conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "data_shape, axes_names, scale_factors, expected, expected_shape",
    [
        # 2D: shape=(4,4), axes ["y","x"], downsample each by 2 → (2,2)
        (
            (4, 4),
            ["y", "x"],
            (2, 2),
            np.array([[2, 4], [10, 12]], dtype=np.uint8),
            (2, 2),
        ),
        # 3D: shape=(1,4,4), axes ["z","y","x"], only Y,X downsample by 2 → (1,2,2)
        (
            (1, 4, 4),
            ["z", "y", "x"],
            (1, 2, 2),
            np.array([[[2, 4], [10, 12]]], dtype=np.uint8),
            (1, 2, 2),
        ),
        # 5D: shape=(1,1,1,4,4), only Y,X downsample by 2 → (1,1,1,2,2)
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            (1, 1, 1, 2, 2),
            np.array([[[[[2, 4], [10, 12]]]]], dtype=np.uint8),
            (1, 1, 1, 2, 2),
        ),
    ],
)
def test_spatial_downsample(
    data_shape: Any,
    axes_names: Any,
    scale_factors: Any,
    expected: Any,
    expected_shape: Any,
) -> None:
    # Create a consecutive array of the given shape
    data = np.arange(int(np.prod(data_shape)), dtype=np.uint8).reshape(data_shape)

    out = spatial_downsample(data, axes_names, scale_factors)
    assert out.shape == expected_shape

    # Round to nearest integer (since mean may produce floats) and compare
    rounded = np.rint(out).astype(expected.dtype)
    np.testing.assert_array_equal(rounded, expected)


@pytest.mark.parametrize(
    "in_shape, axes_names, axes_types, scale_factors, max_levels, expected_levels",
    [
        # 2D: shape=(64,64), downsample halves until 1, but limit to 3 levels
        (
            (64, 64),
            ["y", "x"],
            ["space", "space"],
            (2, 2),
            3,
            [(64, 64), (32, 32), (16, 16)],
        ),
        # 3D: (Z=8,Y=64,X=64), only Y,X halve, limit to 4 levels
        (
            (8, 64, 64),
            ["z", "y", "x"],
            ["space", "space", "space"],
            (2, 2, 2),
            4,
            [(8, 64, 64), (8, 32, 32), (8, 16, 16), (8, 8, 8)],
        ),
        # 4D time: (T=5,Z=32,Y=64,X=64), only Y,X downsample, limit to 2 levels
        (
            (5, 32, 64, 64),
            ["t", "z", "y", "x"],
            ["time", "space", "space", "space"],
            (1, 1, 2, 2),
            2,
            [(5, 32, 64, 64), (5, 32, 32, 32)],
        ),
        # 5D: (1,1,1,4,4), only Y,X downsample, limit to 2 levels
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (1, 1, 1, 2, 2),
            2,
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
        ),
    ],
)
def test_compute_level_shapes_with_max_levels(
    in_shape: Any,
    axes_names: Any,
    axes_types: Any,
    scale_factors: Any,
    max_levels: Any,
    expected_levels: Any,
) -> None:
    writer = OmeZarrWriterV3(
        store=tempfile.mkdtemp(),
        shape=in_shape,
        dtype=np.uint8,
        axes_names=axes_names,
        axes_types=axes_types,
        scale_factors=scale_factors,
        num_levels=None,  # allow full pyramid generation
    )
    # Call _compute_levels with max_levels to limit output
    out = writer._compute_levels(max_levels)
    assert out == expected_levels


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, data_generator, expected_shapes",
    [
        # 2D: (4,4)
        (
            (4, 4),
            None,
            None,
            lambda: np.arange(16, dtype=np.uint8).reshape((4, 4)),
            [(4, 4), (2, 2), (1, 1)],
        ),
        # 3D: (Z=4,Y=8,X=8)
        (
            (4, 8, 8),
            None,
            None,
            lambda: np.random.randint(0, 255, size=(4, 8, 8), dtype=np.uint8),
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
        ),
        # 4D time: (T=3,Z=4,Y=8,X=8)
        (
            (3, 4, 8, 8),
            ["t", "z", "y", "x"],
            ["time", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(3, 4, 8, 8), dtype=np.uint8),
            [(3, 4, 8, 8), (3, 4, 4, 4), (3, 4, 2, 2), (3, 4, 1, 1)],
        ),
        # 4D channel: (C=2,Z=4,Y=8,X=8)
        (
            (2, 4, 8, 8),
            ["c", "z", "y", "x"],
            ["channel", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(2, 4, 8, 8), dtype=np.uint8),
            [(2, 4, 8, 8), (2, 4, 4, 4), (2, 4, 2, 2), (2, 4, 1, 1)],
        ),
        # 5D: (1,1,1,4,4)
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(1, 1, 1, 4, 4), dtype=np.uint8),
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2), (1, 1, 1, 1, 1)],
        ),
    ],
)
def test_write_full_volume_and_metadata(
    shape: Any,
    axes_names: Any,
    axes_types: Any,
    data_generator: Any,
    expected_shapes: Any,
) -> None:
    data = data_generator()
    tmpdir = tempfile.mkdtemp()
    try:
        writer_kwargs = {
            "store": tmpdir,
            "shape": shape,
            "dtype": data.dtype,
            "scale_factors": tuple(2 for _ in shape),
            "num_levels": None,
        }
        if axes_names:
            writer_kwargs["axes_names"] = axes_names
        if axes_types:
            writer_kwargs["axes_types"] = axes_types

        writer = OmeZarrWriterV3(**writer_kwargs)
        writer.write_full_volume(data)

        grp = zarr.open(tmpdir, mode="r")
        for idx, exp_shape in enumerate(expected_shapes):
            arr = grp[str(idx)]
            assert arr.shape == exp_shape

        ome_meta = grp.attrs.asdict()
        validate(ome_meta, version="0.5", model="image", strict=False)

        ms = ome_meta["ome"]["multiscales"][0]
        assert len(ms["datasets"]) == len(expected_shapes)
    finally:
        shutil.rmtree(tmpdir)


@pytest.mark.parametrize(
    "shape, chunks, shards",
    [
        # 2D: (16×16), chunk=(4×4), shard‐factor=(2×2)
        ((16, 16), (4, 4), (2, 2)),
        # 3D: (Z=2, Y=16, X=16), chunk=(1,4,4), shard‐factor=(1,2,2)
        ((2, 16, 16), (1, 4, 4), (1, 2, 2)),
        # 4D time: (T=2, Z=2, Y=16, X=16), chunk=(1,1,4,4), shard‐factor=(1,1,2,2)
        ((2, 2, 16, 16), (1, 1, 4, 4), (1, 1, 2, 2)),
        # 4D channel: (C=3, Z=2, Y=16, X=16), same chunk/shard pattern
        ((3, 2, 16, 16), (1, 1, 4, 4), (1, 1, 2, 2)),
        # 5D: (T=2, C=2, Z=2, Y=16, X=16), chunk=(1,1,1,4,4), shard‐factor=(1,1,1,2,2)
        ((2, 2, 2, 16, 16), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2)),
    ],
)
def test_sharding_and_chunking_applied_to_arrays_high_dim(
    tmp_path: Any,
    shape: Any,
    chunks: Any,
    shards: Any,
) -> None:
    # Zero‐fill an array of the given shape
    data = np.zeros(shape, dtype=np.uint8)
    store = str(tmp_path / "test_highdim.zarr")

    # Instantiate writer with requested chunk sizes and shard factors
    writer = OmeZarrWriterV3(
        store=store,
        shape=shape,
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
    )
    writer.write_full_volume(data)

    grp = zarr.open(store, mode="r")
    for lvl, lvl_shape in enumerate(writer.level_shapes):
        arr = grp[str(lvl)]

        # 1) Verify that arr.chunks was clamped from “chunks”:
        clamped_chunks = writer.chunks[lvl]
        assert arr.chunks == clamped_chunks

        # 2) The on-disk shard‐shape should be (chunk_size * shard_factor)
        shard_factor = writer.shards[lvl]
        assert shard_factor is not None
        expected_shard_shape = tuple(
            clamped_chunks[i] * shard_factor[i] for i in range(len(lvl_shape))
        )
        assert arr.shards == expected_shard_shape


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, axes_units, axes_scale, scale_factors, "
    "channel_kwargs, chunks, shards, filename",
    [
        # Existing TCZYX case (5D)
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            [None, None, None, "micrometer", "micrometer"],
            [1.0, 1.0, 1.0, 0.5, 0.5],
            (1, 1, 1, 2, 2),
            {
                "label": "Ch0",
                "color": "FF0000",
                "active": True,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {"min": 0, "max": 255, "start": 0, "end": 255},
            },
            (1, 1, 1, 4, 4),  # chunks for TCZYX
            (1, 1, 1, 2, 2),  # shards for TCZYX
            "reference_zarr.json",
        ),
        # New TYX case (3D with minimal channel metadata)
        (
            (2, 4, 4),  # (T, Y, X)
            ["t", "y", "x"],
            ["time", "space", "space"],
            [None, "micrometer", "micrometer"],
            [1.0, 0.5, 0.5],
            (1, 2, 2),  # downsample factors for (T, Y, X)
            {
                "label": "Ch0",
                "color": "FF0000",
            },
            (1, 4, 4),  # chunks for TYX
            (1, 2, 2),  # shards for TYX
            "reference_zarr_tyx.json",
        ),
    ],
)
def test_metadata_against_reference(
    tmp_path: Any,
    shape: Any,
    axes_names: Any,
    axes_types: Any,
    axes_units: Any,
    axes_scale: Any,
    scale_factors: Any,
    channel_kwargs: Any,
    chunks: Any,
    shards: Any,
    filename: Any,
) -> None:
    # 1) Create dummy data
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    # 2) Build Channel from the parameterized metadata
    ch0 = Channel(**channel_kwargs)

    # 3) Instantiate writer with all parameters, including the channel
    store_dir = str(tmp_path / "ref_test.zarr")
    writer = OmeZarrWriterV3(
        store=store_dir,
        shape=shape,
        dtype="uint8",
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
        axes_scale=axes_scale,
        scale_factors=scale_factors,
        num_levels=None,
        chunks=chunks,
        shards=shards,
        channels=[ch0],
        creator_info={"name": "pytest", "version": "0.1"},
    )

    # 4) Write out the data
    writer.write_full_volume(data)

    # 5) Load the generated metadata
    grp = zarr.open(store_dir, mode="r")
    generated = grp.attrs.asdict()

    # 6) Load the reference JSON
    uri = LOCAL_RESOURCES_DIR / filename
    with open(uri, "r") as f:
        reference = json.load(f)

    # 7) Compare the metadata dictionaries
    assert generated["ome"] == reference["attributes"]["ome"]


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, scale_factors",
    [
        # (T=2, Y=4, X=4) downsample Y,X by 2
        ((2, 4, 4), ["t", "y", "x"], ["time", "space", "space"], (1, 2, 2)),
        # (T=3, Y=6, X=6) downsample Y,X by 3
        ((3, 6, 6), ["t", "y", "x"], ["time", "space", "space"], (1, 3, 3)),
    ],
)
def test_write_timepoint_vs_full_volume(
    tmp_path: Any,
    shape: Any,
    axes_names: Any,
    axes_types: Any,
    scale_factors: Any,
) -> None:
    """
    Verify that writing each timepoint individually yields identical multiscale
    results to writing the full volume at once.
    """
    # 1) Generate a full 3D array: shape = (T, Y, X)
    full_data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    # 2) Create a writer that writes the full volume
    store_full = str(tmp_path / "full_volume.zarr")
    writer_full = OmeZarrWriterV3(
        store=store_full,
        shape=shape,
        dtype="uint8",
        axes_names=axes_names,
        axes_types=axes_types,
        scale_factors=scale_factors,
        num_levels=None,
    )
    writer_full.write_full_volume(full_data)

    # 3) Create a writer that writes each timepoint separately
    store_tp = str(tmp_path / "timepoints.zarr")
    writer_tp = OmeZarrWriterV3(
        store=store_tp,
        shape=shape,
        dtype="uint8",
        axes_names=axes_names,
        axes_types=axes_types,
        scale_factors=scale_factors,
        num_levels=None,
    )
    # Write each slice at axis 0
    for t in range(shape[0]):
        writer_tp.write_timepoint(t, full_data[t])

    # 4) Open both stores
    grp_full = zarr.open(store_full, mode="r")
    grp_tp = zarr.open(store_tp, mode="r")

    # 5) Compare each level array by array
    for lvl in range(writer_full.num_levels):
        arr_full = grp_full[str(lvl)]
        arr_tp = grp_tp[str(lvl)]
        # The shapes must match
        assert arr_full.shape == arr_tp.shape
        # And the contents must match exactly
        np.testing.assert_array_equal(arr_full[...], arr_tp[...])
