from __future__ import annotations

"""
Dimension-agnostic utilities for building OME-Zarr pyramids.

All helpers in this module are safe for 2–5D shapes. Backward-compatible
entry points and behaviors are preserved for existing callers.
"""

import math
from dataclasses import dataclass
from math import prod
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

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

# ND-friendly tuple type (was fixed 5-tuple before)
DimTuple = Tuple[int, ...]


# (Kept for compatibility; many callers define their own) ----------------------
@dataclass
class ZarrLevel:
    shape: DimTuple
    chunk_size: DimTuple
    dtype: np.dtype
    zarray: zarr.Array


# -----------------------------------------------------------------------------
# Chunking & resizing
# -----------------------------------------------------------------------------


def suggest_chunks(
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    axis_types: List[str],
    target_size: int = 16 << 20,  # ~16 MiB
) -> Tuple[int, ...]:
    """
    Suggest an ND chunk shape aiming for ~``target_size`` bytes per chunk.

    Strategy
    --------
    - Start with 1 along non-spatial axes; fill spatial axes from the budget.
    - For 2–3 spatial dims, distribute the budget across them.
    - If the full array fits in budget, return the full shape.
    """
    itemsize = np.dtype(dtype).itemsize
    max_elems = max(1, target_size // itemsize)

    if prod(shape) <= max_elems:
        return tuple(shape)

    spatial_idxs = [i for i, t in enumerate(axis_types) if t == "space"]
    ndim = len(shape)
    chunk = [1] * ndim

    if len(spatial_idxs) == 0:
        return tuple(chunk)  # nothing to distribute

    # Greedy fill on spatial dims: last dims first (often Y/X)
    remaining = max_elems
    # approximate square/box base for the first spatial axis we fill
    base = int(math.sqrt(max_elems)) if len(spatial_idxs) >= 2 else max_elems
    first = True
    for idx in reversed(spatial_idxs):
        size = shape[idx]
        if first:
            val = min(size, max(1, base))
            first = False
        else:
            val = min(size, max(1, remaining))
        chunk[idx] = val
        remaining = max(1, remaining // max(1, val))

    return tuple(chunk)


def resize(
    image: da.Array, output_shape: Tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    """Block-wise ND resize with Dask + skimage, preserving dtype.

    The function rechunks input to better match the output tiling to avoid
    tiny output chunks, then maps a per-block resize and rechunks back.
    """
    factors = np.array(output_shape) / np.array(image.shape, float)

    # Choose an input chunking that maps neatly to output chunks
    better_chunksize = tuple(
        np.maximum(1, np.ceil(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    # Anticipate each block's output chunk shape
    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    def resize_block(image_block: np.ndarray, block_info: dict) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# Multiscale shape planners
# -----------------------------------------------------------------------------


def compute_level_shapes(
    lvl0shape: Tuple[int, ...],
    scaling: Union[Tuple[float, ...], List[str]],
    nlevels: Union[int, Tuple[int, ...]],
    max_levels: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """
    Compute ND multiscale pyramid level shapes.

    Two compatible calling styles are supported:

    1) Legacy style
       compute_level_shapes((T,C,Z,Y,X), (1,1,1,2,2), 3)
       → applies the numeric scaling to *every* axis.

    2) V3 style (recommended)
       compute_level_shapes(base_shape, axis_names, axis_factors, max_levels)
       → only **Y**/**X** are downsampled; all other axes remain unchanged.
    """
    # V3 style: scaling=list[str], nlevels=tuple[int,...]
    if (
        isinstance(scaling, list)
        and all(isinstance(n, str) for n in scaling)
        and isinstance(nlevels, tuple)
    ):
        axis_names = [n.lower() for n in scaling]
        axis_factors = tuple(int(f) for f in nlevels)
        shapes_v3: List[Tuple[int, ...]] = [tuple(lvl0shape)]

        levels_target = max_levels
        max_autolevels = 64 if levels_target is None else levels_target

        for _ in range(1, max_autolevels):
            prev = shapes_v3[-1]
            nxt: List[int] = []
            for i, size in enumerate(prev):
                name = axis_names[i] if i < len(axis_names) else ""
                fac = axis_factors[i] if i < len(axis_factors) else 1
                if name in ("y", "x") and fac > 1:
                    nxt.append(max(1, size // fac))
                else:
                    nxt.append(size)
            nxt_tuple = tuple(nxt)
            if nxt_tuple == prev:
                break
            shapes_v3.append(nxt_tuple)
            if levels_target is not None and len(shapes_v3) >= levels_target:
                break
        return shapes_v3

    # Legacy style: scaling=tuple[float], nlevels=int
    scaling_factors = cast(Tuple[float, ...], scaling)
    num_levels = cast(int, nlevels)
    shapes_legacy: List[Tuple[int, ...]] = [tuple(lvl0shape)]
    for _ in range(max(1, num_levels) - 1):
        prev = shapes_legacy[-1]
        next_shape = tuple(
            max(int(prev[i] / scaling_factors[i]), 1) for i in range(len(prev))
        )
        shapes_legacy.append(next_shape)
    return shapes_legacy


def get_scale_ratio(
    level0: Tuple[int, ...], level1: Tuple[int, ...]
) -> Tuple[float, ...]:
    """Per-axis scale ratio from level0 to level1 as floats."""
    return tuple(level0[i] / level1[i] for i in range(len(level0)))


def compute_level_chunk_sizes_zslice(
    level_shapes: List[Tuple[int, ...]],
) -> List[DimTuple]:
    """
    Compute ND per-level chunk sizes using a “Z-slice” style policy.

    - **5D (T,C,Z,Y,X)**: reproduce legacy behavior exactly.
    - **2–4D**: full chunk along the last two dims (assumed Y/X), 1 elsewhere,
      and reduce Y/X chunk sizes in lock-step with level downsampling.
    """
    if not level_shapes:
        return []

    ndim = len(level_shapes[0])
    result: List[DimTuple] = []

    if ndim == 5:
        # Legacy exact behavior
        result = [(1, 1, 1, level_shapes[0][3], level_shapes[0][4])]
        for i in range(1, len(level_shapes)):
            prev_shape = level_shapes[i - 1]
            curr_shape = level_shapes[i]
            scale = tuple(prev_shape[j] / curr_shape[j] for j in range(5))
            p = result[i - 1]
            new_chunk: DimTuple = (
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


# -----------------------------------------------------------------------------
# Memory-based chunk sizing
# -----------------------------------------------------------------------------


def chunk_size_from_memory_target(
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    memory_target: int,
    order: Optional[Sequence[str]] = None,
) -> Tuple[int, ...]:
    """
    Suggest an ND chunk shape that fits within ``memory_target`` bytes.

    Policy
    ------
    - If ``order`` is None, we assume the last N of ["T","C","Z","Y","X"].
    - Spatial axes (Z/Y/X) start at full size; others start at 1.
    - Halve all dims until the product fits in the memory target.
    """
    TCZYX = ["T", "C", "Z", "Y", "X"]
    ndim = len(shape)

    # Infer/validate order
    if order is None:
        if ndim <= len(TCZYX):
            order = TCZYX[-ndim:]
        else:
            raise ValueError(f"No default for {ndim}-D shape; pass explicit `order`.")
    elif len(order) != ndim:
        raise ValueError(f"`order` length {len(order)} != shape length {ndim}")

    itemsize = np.dtype(dtype).itemsize

    # Start with full spatial, 1 elsewhere
    chunk_list: List[int] = [
        size if ax.upper() in ("Z", "Y", "X") else 1 for size, ax in zip(shape, order)
    ]

    while int(np.prod(chunk_list)) * itemsize > memory_target:
        chunk_list = [max(s // 2, 1) for s in chunk_list]

    return tuple(chunk_list)


# -----------------------------------------------------------------------------
# Append a new level (v2 helper; still TCZYX-specific due to Reader)
# -----------------------------------------------------------------------------


def add_zarr_level(
    existing_zarr: Union[str, Path],
    scale_factors: Tuple[float, float, float, float, float],  # (T, C, Z, Y, X)
    compressor: Optional[numcodecs.abc.Codec] = None,
    t_batch: int = 4,
) -> None:
    """
    Append one more resolution level to a v2 OME-Zarr, writing in T-slices.

    NOTE: This helper remains TCZYX-specific because ``Reader`` exposes a
    TCZYX-oriented API. If you want a generalized ND version, we can refactor
    ``Reader`` and this function together.
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
