from typing import Any, Dict, Tuple, Union

import dask.array as da
import numpy as np

from .utils import chunk_size_from_memory_target


def get_default_config_for_viz(
    data: Union[np.ndarray, da.Array],
) -> Dict[str, Any]:
    """
    Visualization preset:
      - 3-level XY pyramid (0.5, 0.25 on Y/X)
      - ~16 MiB chunking reused for all levels
      - Let the writer infer axes, zarr_format, image_name, etc.
    """
    shape: Tuple[int, ...] = tuple(data.shape)
    dtype = np.dtype(getattr(data, "dtype", np.uint16))

    # Inline _build_xy_scales
    ndim = len(shape)
    if ndim < 2:
        scale: Tuple[Tuple[float, ...], ...] = tuple()
    else:
        scales_list = []
        for k in (1, 2):  # levels beyond 0
            vec = [1.0] * ndim
            vec[-1] = 2.0 ** (-k)  # X
            vec[-2] = 2.0 ** (-k)  # Y
            scales_list.append(tuple(vec))
        scale = tuple(scales_list)

    chunk_shape = tuple(
        int(x) for x in chunk_size_from_memory_target(shape, dtype, 16 << 20)
    )

    return {
        "shape": shape,
        "dtype": dtype,
        "scale": scale,
        "chunk_shape": chunk_shape,
    }


def get_default_config_for_ml(
    data: Union[np.ndarray, da.Array],
) -> Dict[str, Any]:
    """
    ML preset:
      - Level-0 only (no pyramid)
      - Z-slice chunking (Z=1) when Z exists; ~16 MiB target otherwise
      - Writer infers everything else.
    """
    shape: Tuple[int, ...] = tuple(data.shape)
    dtype = np.dtype(getattr(data, "dtype", np.uint16))

    base_chunk = tuple(
        int(x) for x in chunk_size_from_memory_target(shape, dtype, 16 << 20)
    )

    ndim = len(shape)
    if ndim >= 3:
        z_idx = ndim - 3
        tmp = list(base_chunk)
        tmp[z_idx] = 1
        chunk_shape = tuple(int(x) for x in tmp)
    else:
        chunk_shape = base_chunk

    return {
        "shape": shape,
        "dtype": dtype,
        "scale": tuple(),  # no extra levels
        "chunk_shape": chunk_shape,
    }
