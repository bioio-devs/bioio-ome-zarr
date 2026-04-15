from typing import Any, Dict, List, Tuple, Union

import dask.array as da
import numpy as np

from .utils import multiscale_chunk_size_from_memory_target

_CHUNK_TARGET_BYTES = 16 << 20  # 16 MiB


def _pyramid_level_shapes(
    level0_shape: Tuple[int, ...], n_spatial: int
) -> List[Tuple[int, ...]]:
    """
    Build a 3-level pyramid from level-0, halving the last n_spatial axes at each level.
    Non-spatial axes are unchanged. Shapes are floored with a minimum of 1.
    If ndim < n_spatial, all axes are treated as spatial.
    """
    ndim = len(level0_shape)
    spatial_indices = list(range(ndim - min(n_spatial, ndim), ndim))

    def downsample(power_of_two: int) -> Tuple[int, ...]:
        factor = 2**power_of_two
        mutable = list(level0_shape)
        for i in spatial_indices:
            mutable[i] = max(1, int(level0_shape[i]) // factor)
        return tuple(int(x) for x in mutable)

    return [
        tuple(int(x) for x in level0_shape),  # level 0
        downsample(1),  # level 1 (÷2)
        downsample(2),  # level 2 (÷4)
    ]


def get_default_config_for_viz(
    data: Union[np.ndarray, da.Array],
    downsample_z: bool = False,
) -> Dict[str, Any]:
    """
    Visualization preset:
      - 3-level pyramid (levels 0/1/2 with spatial axes ÷1, ÷2, ÷4)
      - downsample_z=False (default): halve Y/X only (2D viewers, e.g. napari)
      - downsample_z=True: halve Z/Y/X equally (3D viewers, e.g. Neuroglancer)
      - ~16 MiB chunking suggested from level-0, reused for all levels
      - Writer infers axes, zarr_format, image_name, etc.
    """
    level0_shape: Tuple[int, ...] = tuple(int(x) for x in data.shape)
    dtype = np.dtype(getattr(data, "dtype", np.uint16))

    n_spatial = 3 if downsample_z else 2
    level_shapes = _pyramid_level_shapes(level0_shape, n_spatial)

    # One chunk shape applied to all levels (writer will replicate it)
    chunk_shape = tuple(
        int(x)
        for x in multiscale_chunk_size_from_memory_target(
            [level0_shape], dtype, _CHUNK_TARGET_BYTES
        )[0]
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
        int(x)
        for x in multiscale_chunk_size_from_memory_target(
            [level0_shape], dtype, _CHUNK_TARGET_BYTES
        )[0]
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
