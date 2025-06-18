import json
import shutil
import tempfile
from typing import Any, List, Tuple

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr.validate import validate

from bioio_ome_zarr.writers import Channel, OmeZarrWriterV3

from ..conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, data_generator, expected_shapes",
    [
        (
            (4, 4),
            None,
            None,
            lambda: np.arange(16, dtype=np.uint8).reshape((4, 4)),
            [(4, 4), (2, 2), (1, 1)],
        ),
        (
            (4, 8, 8),
            None,
            None,
            lambda: np.random.randint(0, 255, size=(4, 8, 8), dtype=np.uint8),
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
        ),
        (
            (3, 4, 8, 8),
            ["t", "z", "y", "x"],
            ["time", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(3, 4, 8, 8), dtype=np.uint8),
            [(3, 4, 8, 8), (3, 4, 4, 4), (3, 4, 2, 2), (3, 4, 1, 1)],
        ),
        (
            (2, 4, 8, 8),
            ["c", "z", "y", "x"],
            ["channel", "space", "space", "space"],
            lambda: np.random.randint(0, 255, size=(2, 4, 8, 8), dtype=np.uint8),
            [(2, 4, 8, 8), (2, 4, 4, 4), (2, 4, 2, 2), (2, 4, 1, 1)],
        ),
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
    shape: Tuple[int, ...],
    axes_names: Any,
    axes_types: Any,
    data_generator: Any,
    expected_shapes: List[Tuple[int, ...]],
) -> None:
    tmpdir = tempfile.mkdtemp()
    try:
        # Arrange
        data = data_generator()
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

        # Act
        writer.write_full_volume(data)

        # Assert: check shapes and metadata
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
    "shape, chunk_size, shard_factor",
    [
        ((16, 16), (4, 4), (2, 2)),
        ((2, 16, 16), (1, 4, 4), (1, 2, 2)),
        ((2, 2, 16, 16), (1, 1, 4, 4), (1, 1, 2, 2)),
        ((3, 2, 16, 16), (1, 1, 4, 4), (1, 1, 2, 2)),
        ((2, 2, 2, 16, 16), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2)),
    ],
)
def test_sharding_and_chunking_applied_to_arrays_high_dim(
    tmp_path: Any,
    shape: Tuple[int, ...],
    chunk_size: Tuple[int, ...],
    shard_factor: Tuple[int, ...],
) -> None:
    # Arrange
    data = np.zeros(shape, dtype=np.uint8)
    store = str(tmp_path / "test_highdim.zarr")
    writer = OmeZarrWriterV3(
        store=store,
        shape=shape,
        dtype=data.dtype,
        scale_factors=(1, 1, 1, 2, 2),
        chunk_size=chunk_size,
        shard_factor=shard_factor,
    )

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(store, mode="r")
    for lvl, lvl_shape in enumerate(writer.level_shapes):
        arr = grp[str(lvl)]
        # chunk_size is uniform
        assert arr.chunks == writer.chunk_size

        # shard_factor is uniform
        if writer.shard_factor is not None:
            expected_shard_factor = tuple(
                writer.chunk_size[i] * writer.shard_factor[i]
                for i in range(len(lvl_shape))
            )
            assert arr.shards == expected_shard_factor


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, axes_units, axes_scale, scale_factors, "
    "channel_kwargs, chunk_size, shard_factor, filename",
    [
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
            (1, 1, 1, 4, 4),
            (1, 1, 1, 2, 2),
            "reference_zarr.json",
        ),
        (
            (2, 4, 4),
            ["t", "y", "x"],
            ["time", "space", "space"],
            [None, "micrometer", "micrometer"],
            [1.0, 0.5, 0.5],
            (1, 2, 2),
            {"label": "Ch0", "color": "FF0000"},
            (1, 4, 4),
            (1, 2, 2),
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
    chunk_size: Any,
    shard_factor: Any,
    filename: Any,
) -> None:
    # Arrange
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    ch0 = Channel(**channel_kwargs)
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
        chunk_size=chunk_size,
        shard_factor=shard_factor,
        channels=[ch0],
        creator_info={"name": "pytest", "version": "0.1"},
    )

    # Act
    writer.write_full_volume(data)

    # Assert
    grp = zarr.open(store_dir, mode="r")
    generated = grp.attrs.asdict()
    uri = LOCAL_RESOURCES_DIR / filename
    with open(uri, "r") as f:
        reference = json.load(f)
    assert generated["ome"] == reference["attributes"]["ome"]


@pytest.mark.parametrize(
    "shape, axes_names, axes_types, factors, chunk_size, shard_factor",
    [
        (
            (2, 4, 4),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 2, 2),
            (1, 2, 2),
            (1, 1, 1),
        ),
        (
            (3, 6, 6),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 3, 3),
            (1, 2, 2),
            (1, 1, 1),
        ),
        (
            (2, 128, 128),
            ["t", "y", "x"],
            ["time", "space", "space"],
            (1, 4, 4),
            (1, 32, 32),
            (1, 2, 2),
        ),
        (
            (2, 2, 4, 128, 128),
            ["t", "c", "z", "y", "x"],
            ["time", "channel", "space", "space", "space"],
            (1, 1, 2, 4, 4),
            (1, 1, 2, 32, 32),
            (1, 1, 1, 2, 2),
        ),
    ],
)
def test_full_vs_timepoint_equivalence(
    tmp_path: Any,
    shape: Tuple[int, ...],
    axes_names: List[str],
    axes_types: List[str],
    factors: Tuple[int, ...],
    chunk_size: Tuple[int, ...],
    shard_factor: Tuple[int, ...],
) -> None:
    """
    Writing full volume vs per-timepoint yields identical multiscale Zarrs,
    preserving data, chunk, and shard layouts, including axis metadata.
    """
    # Arrange
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    full_store = str(tmp_path / "full.zarr")
    tp_store = str(tmp_path / "tp.zarr")

    w_full = OmeZarrWriterV3(
        store=full_store,
        shape=shape,
        dtype=data.dtype,
        axes_names=axes_names,
        axes_types=axes_types,
        scale_factors=factors,
        num_levels=None,
        chunk_size=chunk_size,
        shard_factor=shard_factor,
    )
    w_tp = OmeZarrWriterV3(
        store=tp_store,
        shape=shape,
        dtype=data.dtype,
        axes_names=axes_names,
        axes_types=axes_types,
        scale_factors=factors,
        num_levels=None,
        chunk_size=chunk_size,
        shard_factor=shard_factor,
    )

    # Act
    w_full.write_full_volume(data)
    for t in range(shape[0]):
        slice_data = data[t]
        slice_da = da.from_array(slice_data, chunks=chunk_size[1:])
        w_tp.write_timepoint(t, slice_da)

    # Assert
    grp_full = zarr.open(full_store, mode="r")
    grp_tp = zarr.open(tp_store, mode="r")
    for lvl, _ in enumerate(w_full.level_shapes):
        arr_full = grp_full[str(lvl)]
        arr_tp = grp_tp[str(lvl)]

        # 1) Data equality
        np.testing.assert_array_equal(arr_full[...], arr_tp[...])

        # 2) Chunk layout
        assert arr_full.chunks == chunk_size
        assert arr_tp.chunks == chunk_size

        # 3) Shard layout
        expected_shard = tuple(c * s for c, s in zip(chunk_size, shard_factor))
        assert arr_full.shards == expected_shard
        assert arr_tp.shards == expected_shard
