from typing import Any, Dict, List, Tuple, Union

import dask.array as da
import numpy as np

from .utils import chunk_size_from_memory_target


def _xy_pyramid_level_shapes(level0_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Build a 3-level XY pyramid from level-0: halve Y/X at levels 1 and 2.
    Non-spatial axes are unchanged. Shapes are floored with a minimum of 1.
    """
    ndim = len(level0_shape)
    if ndim < 2:
        return [tuple(int(x) for x in level0_shape)]

    y_idx = ndim - 2
    x_idx = ndim - 1

    def downsample(power_of_two: int) -> Tuple[int, ...]:
        factor = 2**power_of_two
        mutable = list(level0_shape)
        mutable[y_idx] = max(1, int(level0_shape[y_idx]) // factor)
        mutable[x_idx] = max(1, int(level0_shape[x_idx]) // factor)
        return tuple(int(x) for x in mutable)

    return [
        tuple(int(x) for x in level0_shape),  # level 0
        downsample(1),  # level 1 (÷2 on Y/X)
        downsample(2),  # level 2 (÷4 on Y/X)
    ]


def get_default_config_for_viz(
    data: Union[np.ndarray, da.Array],
) -> Dict[str, Any]:
    """
    Visualization preset:
      - 3-level XY pyramid (levels 0/1/2 with Y/X ÷1, ÷2, ÷4)
      - ~16 MiB chunking suggested from level-0, reused for all levels
      - Writer infers axes, zarr_format, image_name, etc.
    """
    level0_shape: Tuple[int, ...] = tuple(int(x) for x in data.shape)
    dtype = np.dtype(getattr(data, "dtype", np.uint16))

    level_shapes = _xy_pyramid_level_shapes(level0_shape)

    # One chunk shape applied to all levels (writer will replicate it)
    chunk_shape = tuple(
        int(x) for x in chunk_size_from_memory_target(level0_shape, dtype, 16 << 20)
    )

    return {
        "level_shapes": level_shapes,
        "dtype": dtype,
        "chunk_shape": chunk_shape,
    }


def get_default_config_for_ml(
    data: Union[np.ndarray, da.Array],
) -> Dict[str, Any]:
    """
    ML preset:
      - Level-0 only (no pyramid)
      - Prefer Z-slice chunking (Z=1) when Z exists; else ~16 MiB target
      - Writer infers remaining fields.
    """
    level0_shape: Tuple[int, ...] = tuple(int(x) for x in data.shape)
    dtype = np.dtype(getattr(data, "dtype", np.uint16))

    base_chunk = tuple(
        int(x) for x in chunk_size_from_memory_target(level0_shape, dtype, 16 << 20)
    )

    # If we have at least (… Z Y X), set Z chunk to 1
    if len(level0_shape) >= 3:
        z_idx = len(level0_shape) - 3
        chunk_list = list(base_chunk)
        chunk_list[z_idx] = 1
        chunk_shape = tuple(int(x) for x in chunk_list)
    else:
        chunk_shape = base_chunk

    return {
        "level_shapes": [level0_shape],
        "dtype": dtype,
        "chunk_shape": chunk_shape,
    }
