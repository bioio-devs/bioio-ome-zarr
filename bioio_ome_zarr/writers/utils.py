import math
from dataclasses import dataclass
from math import prod
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import dask.array as da
import numpy as np
import skimage.transform
import zarr

DimTuple = Tuple[int, int, int, int, int]


@dataclass
class ZarrLevel:
    shape: DimTuple
    chunk_size: DimTuple
    dtype: np.dtype
    zarray: zarr.Array


def suggest_chunks(
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    axis_types: List[str],
    target_size: int = 16 << 20,  # 16 MB
) -> Tuple[int, ...]:
    """
    Suggest chunk shapes aiming for ~target_size bytes per chunk.
    """
    itemsize = np.dtype(dtype).itemsize
    maxe = target_size // itemsize
    # if the byte-budget covers the full array, use the entire shape
    if prod(shape) <= maxe:
        return tuple(shape)
    spatial_idxs = [i for i, t in enumerate(axis_types) if t == "space"]
    ndim = len(shape)
    chunk = [1] * ndim
    if len(spatial_idxs) in (2, 3):
        remaining = maxe
        base = int(math.sqrt(maxe))
        first = True
        for idx in reversed(spatial_idxs):
            size = shape[idx]
            if first:
                val = min(size, base)
                first = False
            else:
                val = min(size, max(1, remaining))
            chunk[idx] = val
            remaining //= val
    return tuple(chunk)


def resize(
    image: da.Array, output_shape: Tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    factors = np.array(output_shape) / np.array(image.shape, float)
    better_chunksize = tuple(
        np.maximum(1, np.ceil(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        chunk_output_shape = tuple(
            np.ceil(np.array(image_block.shape) * factors).astype(int)
        )
        return skimage.transform.resize(
            image_block, chunk_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)


def compute_level_shapes(
    lvl0shape: Tuple[int, ...],
    scaling: Union[Tuple[float, ...], List[str]],
    nlevels: Union[int, Tuple[int, ...]],
    max_levels: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """
    Compute multiscale pyramid level shapes.

    Supports two signatures:
      - Legacy: (lvl0shape, scaling: Tuple[float,...], nlevels: int)
      - V3:     (base_shape, axis_names: List[str],
                axis_factors: Tuple[int,...], max_levels: int)
    """
    # V3 mode: scaling is list of axis names, nlevels is tuple of int factors
    if (
        isinstance(scaling, list)
        and all(isinstance(n, str) for n in scaling)
        and isinstance(nlevels, tuple)
    ):
        axis_names = [n.lower() for n in scaling]
        axis_factors = nlevels
        shapes: List[Tuple[int, ...]] = [tuple(lvl0shape)]
        lvl = 1
        while max_levels is None or lvl < (max_levels or 0):
            prev = shapes[-1]
            nxt: List[int] = []
            for i, size in enumerate(prev):
                name = axis_names[i]
                factor = axis_factors[i]
                if name in ("x", "y") and factor > 1:
                    nxt.append(max(1, size // factor))
                else:
                    nxt.append(size)
            nxt_tuple = tuple(nxt)
            if nxt_tuple == prev:
                break
            shapes.append(nxt_tuple)
            lvl += 1
        return shapes
    # Legacy mode: scaling is tuple of floats, nlevels is int
    scaling_factors = cast(Tuple[float, ...], scaling)
    num_levels = cast(int, nlevels)
    # Reuse the same variable 'shapes' without re-annotation
    shapes = [tuple(lvl0shape)]
    for _ in range(num_levels - 1):
        prev = shapes[-1]
        next_shape = tuple(
            max(int(prev[i] / scaling_factors[i]), 1) for i in range(len(prev))
        )
        shapes.append(next_shape)
    return shapes


def get_scale_ratio(
    level0: Tuple[int, ...], level1: Tuple[int, ...]
) -> Tuple[float, ...]:
    return tuple(level0[i] / level1[i] for i in range(len(level0)))


def compute_level_chunk_sizes_zslice(
    level_shapes: List[Tuple[int, ...]]
) -> List[DimTuple]:
    """
    Compute Z-slice–based chunk sizes for a multiscale pyramid.

    Parameters
    ----------
    level_shapes : List[Tuple[int, ...]]
        Series of level shapes (potentially N-dimensional),
        but expecting at least 5 dimensions for TCZYX indexing.

    Returns
    -------
    List[DimTuple]
        Chunk sizes as 5-tuples (T, C, Z, Y, X).
    """
    # Initialize with top level: full Y/X, single Z,C,T
    chunk_sizes: List[DimTuple] = [(1, 1, 1, level_shapes[0][3], level_shapes[0][4])]
    for i in range(1, len(level_shapes)):
        prev_shape = level_shapes[i - 1]
        curr_shape = level_shapes[i]
        # Compute per-axis scale ratios
        scale = tuple(prev_shape[j] / curr_shape[j] for j in range(len(prev_shape)))
        prev_chunk = chunk_sizes[i - 1]
        new: DimTuple = (
            1,
            1,
            int(scale[4] * scale[3] * prev_chunk[2]),
            int(prev_chunk[3] / scale[3]),
            int(prev_chunk[4] / scale[4]),
        )
        chunk_sizes.append(new)
    return chunk_sizes


def chunk_size_from_memory_target(
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    memory_target: int,
    order: Optional[Sequence[str]] = None,
) -> Tuple[int, ...]:
    """
    Suggest a chunk shape that fits within `memory_target` bytes.

    - If `order` is None, assume the last N of ["T","C","Z","Y","X"].
    - Spatial axes (Z/Y/X) start at full size; others start at 1.
    - Halve all dims until under the target.
    """
    TCZYX = ["T", "C", "Z", "Y", "X"]
    ndim = len(shape)

    # Infer or validate axis ordering
    if order is None:
        if ndim <= len(TCZYX):
            order = TCZYX[-ndim:]
        else:
            raise ValueError(f"No default for {ndim}-D shape; pass explicit `order`")
    elif len(order) != ndim:
        raise ValueError(f"`order` length {len(order)} != shape length {ndim}")

    # Compute item size in bytes
    itemsize = np.dtype(dtype).itemsize

    # Build a mutable list of initial chunk sizes
    chunk_list: List[int] = [
        size if ax.upper() in ("Z", "Y", "X") else 1 for size, ax in zip(shape, order)
    ]

    # Halve dims until within memory target
    while np.prod(chunk_list) * itemsize > memory_target:
        chunk_list = [max(s // 2, 1) for s in chunk_list]

    # Return as an immutable tuple
    return tuple(chunk_list)
