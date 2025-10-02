import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import dask.array as da
import numcodecs
import numpy as np
import skimage.transform
import zarr
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.storage import LocalStore

from bioio_ome_zarr.reader import Reader

DimSeq = Sequence[int]
PerLevelDimSeq = Sequence[DimSeq]

# LEGACY (Remove with V2 writer)
DimTuple = Tuple[int, ...]


@dataclass
class ZarrLevel:
    """Descriptor for a Zarr multiscale level."""

    shape: DimSeq
    chunk_size: DimSeq
    dtype: np.dtype
    zarray: zarr.Array


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


# LEGACY (Remove with V2 writer)
def get_scale_ratio(
    level0: Tuple[int, ...], level1: Tuple[int, ...]
) -> Tuple[float, ...]:
    return tuple(level0[i] / level1[i] for i in range(len(level0)))


# LEGACY (Remove with V2 writer)
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


# LEGACY (Remove with V2 writer)
def compute_level_chunk_sizes_zslice(
    level_shapes: List[Tuple[int, ...]],
) -> List[DimSeq]:
    """
    Compute Z-slice–based chunk sizes for a multiscale pyramid.
    Parameters
    ----------
    level_shapes : List[Tuple[int, ...]]
        Series of level shapes (potentially N-dimensional),
        but expecting at least 5 dimensions for TCZYX indexing.
    Returns
    -------
    List[DimSeq]
        Chunk sizes as 5-tuples (T, C, Z, Y, X).
    """

    ndim = len(level_shapes[0])
    result: List[DimSeq] = []

    if ndim == 5:
        # Legacy exact behavior
        result = [(1, 1, 1, level_shapes[0][3], level_shapes[0][4])]
        for i in range(1, len(level_shapes)):
            prev_shape = level_shapes[i - 1]
            curr_shape = level_shapes[i]
            scale = tuple(prev_shape[j] / curr_shape[j] for j in range(5))
            p = result[i - 1]
            new_chunk: DimSeq = (
                1,
                1,
                int(scale[4] * scale[3] * p[2]),
                max(1, int(p[3] / max(1, scale[3]))),
                max(1, int(p[4] / max(1, scale[4]))),
            )
            result.append(new_chunk)
        return result

    # Generic 2–4D path (assume last two dims are spatial Y, X)
    y_idx = max(0, ndim - 2)
    x_idx = max(0, ndim - 1)

    first = [1] * ndim
    first[y_idx] = level_shapes[0][y_idx]
    first[x_idx] = level_shapes[0][x_idx]
    result = [tuple(first)]

    for i in range(1, len(level_shapes)):
        prev_shape = level_shapes[i - 1]
        curr_shape = level_shapes[i]
        prev_chunk = list(result[-1])

        y_scale = max(1, int(prev_shape[y_idx] / max(1, curr_shape[y_idx])))
        x_scale = max(1, int(prev_shape[x_idx] / max(1, curr_shape[x_idx])))

        prev_chunk[y_idx] = max(1, int(prev_chunk[y_idx] / y_scale))
        prev_chunk[x_idx] = max(1, int(prev_chunk[x_idx] / x_scale))
        result.append(tuple(prev_chunk))

    return result


# LEGACY (Remove with V3 writer)
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
    while int(np.prod(chunk_list)) * itemsize > memory_target:
        chunk_list = [max(s // 2, 1) for s in chunk_list]
    # Return as an immutable tuple
    return tuple(chunk_list)


def add_zarr_level(
    existing_zarr: Union[str, Path],
    scale_factors: Tuple[float, float, float, float, float],  # (T, C, Z, Y, X)
    compressor: Optional[numcodecs.abc.Codec] = None,
    t_batch: int = 4,
) -> None:
    """
    Append one more resolution level to an OME-Zarr, writing in T-slices.
    """
    rdr = Reader(existing_zarr)
    levels = list(rdr.resolution_levels)
    if not levels:
        raise RuntimeError("No existing resolution levels found.")

    src_idx = max(levels)
    src_shape = rdr.resolution_level_dims[src_idx]
    dtype = rdr.dtype

    new_shape = tuple(int(np.ceil(s * f)) for s, f in zip(src_shape, scale_factors))
    chunks = chunk_size_from_memory_target(new_shape, dtype, 16 * 1024 * 1024)
    store = LocalStore(str(existing_zarr))
    group = zarr.open_group(store=store, mode="a", zarr_format=2)
    new_idx = src_idx + 1
    arr = group.create_array(
        name=str(new_idx),
        shape=new_shape,
        chunks=chunks,
        dtype=dtype,
        compressors=[compressor] if compressor is not None else None,
        overwrite=False,
        fill_value=0,
        chunk_key_encoding=V2ChunkKeyEncoding(separator="/").to_dict(),
    )

    total_t = src_shape[0]
    for t_start in range(0, total_t, t_batch):
        t_end = min(t_start + t_batch, total_t)
        t_block = rdr.get_image_dask_data(
            "TCZYX", resolution_level=src_idx, T=slice(t_start, t_end)
        )
        resized = resize(t_block, (t_end - t_start, *new_shape[1:]), order=0).astype(
            dtype
        )

        da.to_zarr(
            resized,
            arr,
            region=(slice(t_start, t_end),) + (slice(None),) * (resized.ndim - 1),
            overwrite=True,
        )

    ms = group.attrs.get("multiscales", [{}])
    datasets = ms[0].setdefault("datasets", [])
    if datasets:
        last_scale = datasets[-1]["coordinateTransformations"][0]["scale"]
    else:
        last_scale = [1] * len(scale_factors)
    new_scale = [p * f for p, f in zip(last_scale, scale_factors)]

    datasets.append(
        {
            "path": str(new_idx),
            "coordinateTransformations": [
                {"type": "scale", "scale": new_scale},
                {"type": "translation", "translation": [0] * len(scale_factors)},
            ],
        }
    )
    group.attrs["multiscales"] = ms

    print(
        f"Added level {new_idx}: shape={new_shape}, chunks={chunks}, scale={new_scale}"
    )


def multiscale_chunk_size_from_memory_target(
    level_shapes: Sequence[Sequence[int]],
    dtype: Union[str, np.dtype],
    memory_target: int,
) -> List[Sequence[int]]:
    """
    Compute per-level chunk shapes under a fixed byte budget, **prioritizing the
    highest-index axis first** (i.e., grow X, then Y, then Z, ... moving left).

    Note
    -----
    These chunk sizes represent an **in-memory target only** and do not account
    for compression.

    Returns
    -------
    list[Sequence[int]]
        Per-level chunk shapes (same ndim/order as the input).
    """
    if not level_shapes:
        raise ValueError("level_shapes cannot be empty")

    # Validation
    for i, shp in enumerate(level_shapes):
        if len(shp) < 2:
            raise ValueError(f"level_shapes[{i}] must have ndim >= 2, got {len(shp)}")
        if any(int(d) < 1 for d in shp):
            raise ValueError(f"level_shapes[{i}] has non-positive dimension(s): {shp}")

    itemsize = np.dtype(dtype).itemsize
    if memory_target < itemsize:
        raise ValueError(f"memory_target {memory_target} < dtype size {itemsize}")

    budget_elems = max(1, memory_target // itemsize)

    out: List[Sequence[int]] = []
    for shp in level_shapes:
        # Start with all-ones, grow from rightmost axis leftward
        chunk = [1] * len(shp)
        for axis in reversed(range(len(shp))):
            used = math.prod(chunk)
            if used >= budget_elems:
                # No room left; keep remaining (left) axes at 1
                break
            max_here = budget_elems // used
            # Cap by the level dimension and ensure at least 1
            chunk[axis] = max(1, min(int(shp[axis]), int(max_here)))
        out.append(tuple(chunk))

    return out
