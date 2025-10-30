from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import zarr

from bioio_ome_zarr.writers import OMEZarrWriter

VALID_LEVELS = [(1, 8, 8), (1, 4, 4)]


@pytest.mark.parametrize(
    "zarr_format, level_shapes, chunk_shape, shard_shape, match",
    [
        # ---------------- Structural validation ----------------
        # Empty level_shapes should fail
        (3, [], None, None, r"level_shapes cannot be empty"),
        # Per-level ndim mismatch
        (
            3,
            [(1, 8, 8), (1, 4)],
            None,
            None,
            r"level_shapes\[1] length 2 != ndim 3",
        ),
        # Empty chunk_shape is invalid when explicitly provided
        (3, VALID_LEVELS, [], None, r"chunk_shape cannot be empty"),
        # Chunk ndim mismatch
        (3, VALID_LEVELS, (4, 4), None, r"chunk_shape length 2 != ndim 3"),
        # Chunk per-level count mismatch
        (
            3,
            VALID_LEVELS,
            [(1, 4, 4), (1, 4, 4), (1, 4, 4)],
            None,
            r"chunk_shape must have 2 entries \(per level\), got 3",
        ),
        # Empty shard_shape is invalid when explicitly provided
        (3, VALID_LEVELS, None, [], r"shard_shape cannot be empty"),
        # Shard ndim mismatch
        (
            3,
            VALID_LEVELS,
            None,
            [(2, 2), (2, 2)],
            r"shard_shape\[0] length 2 != ndim 3",
        ),
        # Shard per-level count mismatch
        (
            3,
            VALID_LEVELS,
            None,
            [(1, 2, 2)],
            r"shard_shape must have 2 entries \(per level\), got 1",
        ),
        # ---------------- Chunk validation ----------------
        # Chunk dimension must be >= 1
        (
            3,
            VALID_LEVELS,
            [(1, 0, 4), (1, 2, 2)],
            None,
            r"chunk_shape\[0]\[1] must be >= 1",
        ),
        # Per-level count mismatch after normalization
        (
            3,
            [(1, 8, 8), (1, 4, 4), (1, 2, 2)],
            [(1, 4, 4), (1, 2, 2)],
            None,
            r"chunk_shape must have 3 entries",
        ),
        # ---------------- Shard validation ----------------
        # Sharding not supported for Zarr v2
        (
            2,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 2), (1, 2, 2)],
            r"shard_shape is not supported for Zarr v2",
        ),
        # Shard per-level count mismatch
        (
            3,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 2)],
            r"shard_shape must have 2 entries",
        ),
        # Shard ndim mismatch
        (
            3,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 2), (2, 2)],
            r"shard_shape\[1] length 2 != ndim 3",
        ),
        # Shard dimension must be >= 1
        (
            3,
            VALID_LEVELS,
            [(1, 2, 2), (1, 2, 2)],
            [(1, 2, 0), (1, 2, 2)],
            r"shard_shape\[0]\[2] must be >= 1",
        ),
        # Shard must be a multiple of the chunk size (level 0)
        (
            3,
            VALID_LEVELS,
            [(1, 4, 4), (1, 2, 2)],
            [(1, 6, 8), (1, 2, 2)],
            r"must be a multiple of chunk_dim 4",
        ),
        # Shard must be a multiple of the chunk size (level 1)
        (
            3,
            VALID_LEVELS,
            [(1, 4, 4), (1, 2, 2)],
            [(1, 8, 8), (1, 3, 2)],
            r"must be a multiple of chunk_dim 2",
        ),
    ],
)
def test_writer_validation_errors(
    zarr_format: int,
    level_shapes: list[tuple[int, ...]],
    chunk_shape: list[tuple[int, ...]] | tuple[int, ...] | None,
    shard_shape: list[tuple[int, ...]] | None,
    match: str,
) -> None:
    """Ensure invalid configurations raise the expected ValueError."""

    kwargs = dict(
        store="in-memory.zarr",  # no disk write called in init.
        dtype=np.uint8,
        axes_names=["t", "y", "x"],
    )
    kwargs["zarr_format"] = zarr_format

    def build() -> OMEZarrWriter:
        return OMEZarrWriter(
            level_shapes=level_shapes,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            **kwargs,
        )

    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    "level_shapes, axes_names, data, match",
    [
        # shape mismatch: T differs
        (
            [(2, 8, 8), (1, 4, 4)],
            ["t", "y", "x"],
            np.zeros((1, 8, 8), dtype=np.uint8),
            r"write_full_volume: input shape does not match level-0 shape\. "
            r"Got \(1, 8, 8\) vs expected \(2, 8, 8\)",
        ),
        # shape mismatch: Y differs
        (
            [(2, 8, 8), (1, 4, 4)],
            ["t", "y", "x"],
            np.zeros((2, 7, 8), dtype=np.uint8),
            r"write_full_volume: input shape does not match level-0 shape\. "
            r"Got \(2, 7, 8\) vs expected \(2, 8, 8\)",
        ),
    ],
)
def test_write_full_volume_validation_errors(
    level_shapes: List[Tuple[int, ...]],
    axes_names: List[str],
    data: np.ndarray,
    match: str,
) -> None:
    writer = OMEZarrWriter(
        store=zarr.storage.MemoryStore(),
        level_shapes=level_shapes,
        dtype=np.uint8,
        axes_names=axes_names,
        zarr_format=3,
    )
    with pytest.raises(ValueError, match=match):
        writer.write_full_volume(data)


@pytest.mark.parametrize(
    "axes_names, level_shapes, data, kwargs, match",
    [
        # No T axis
        (
            ["y", "x"],
            [(8, 8), (4, 4)],
            np.zeros((8, 8), dtype=np.uint8),
            dict(),
            r"write_timepoints\(\) requires a 'T' axis",
        ),
        # ndim mismatch
        (
            ["t", "y", "x"],
            [(3, 8, 8), (3, 4, 4)],
            np.zeros((8, 8), dtype=np.uint8),
            dict(),
            r"write_timepoints: array ndim \(2\) must match writer\.ndim \(3\)",
        ),
        # non-T dims mismatch (Y)
        (
            ["t", "y", "x"],
            [(3, 8, 8), (3, 4, 4)],
            np.zeros((3, 7, 8), dtype=np.uint8),
            dict(),
            r"non-T axes must match destination level-0 shape\. Axis 1: "
            r"got 7, expected 8",
        ),
        # start_T_src out of range (>= src_T)
        (
            ["t", "y", "x"],
            [(3, 8, 8), (3, 4, 4)],
            np.zeros((3, 8, 8), dtype=np.uint8),
            dict(start_T_src=3, start_T_dest=0),
            r"start_T_src \(3\) out of range \[0, 3\)",
        ),
        # start_T_dest out of range (>= dst_T)
        (
            ["t", "y", "x"],
            [(2, 8, 8), (2, 4, 4)],
            np.zeros((2, 8, 8), dtype=np.uint8),
            dict(start_T_src=0, start_T_dest=2),
            r"start_T_dest \(2\) out of range \[0, 2\)",
        ),
        # total_T <= 0
        (
            ["t", "y", "x"],
            [(3, 8, 8), (3, 4, 4)],
            np.zeros((3, 8, 8), dtype=np.uint8),
            dict(total_T=0),
            r"total_T must be > 0",
        ),
        # total_T > src_avail
        (
            ["t", "y", "x"],
            [(3, 8, 8), (3, 4, 4)],
            np.zeros((3, 8, 8), dtype=np.uint8),
            dict(start_T_src=1, start_T_dest=0, total_T=3),
            r"requested total_T exceeds available source timepoints",
        ),
        # total_T > dst_avail
        (
            ["t", "y", "x"],
            [(2, 8, 8), (2, 4, 4)],
            np.zeros((2, 8, 8), dtype=np.uint8),
            dict(start_T_src=0, start_T_dest=1, total_T=2),
            r"requested total_T exceeds available destination space",
        ),
    ],
)
def test_write_timepoints_validation_errors(
    axes_names: List[str],
    level_shapes: List[Tuple[int, ...]],
    data: np.ndarray,
    kwargs: Dict[str, Any],
    match: str,
) -> None:
    writer = OMEZarrWriter(
        store=zarr.storage.MemoryStore(),
        level_shapes=level_shapes,
        dtype=np.uint8,
        axes_names=axes_names,
        zarr_format=3,
    )
    with pytest.raises(ValueError, match=match):
        writer.write_timepoints(data, **kwargs)
