from typing import Any, Dict, Union

import dask.array as da
import numpy as np

from .utils import chunk_size_from_memory_target


def get_ome_zarr_writer_config_for_viz(
    data: Union[np.ndarray, da.Array],
) -> Dict[str, Any]:
    """
    Generate a default OmeZarrWriterV3 config for visualization use.

    Parameters
    ----------
    data : Union[np.ndarray, dask.array.Array]
        Input data to inspect (e.g. shape TCZYX or similar).

    Returns
    -------
    config : Dict[str, Any]
        A config dictionary to pass into `OmeZarrWriterV3`.
        Customize fields after calling this function.

    Example
    -------
    >>> import numpy as np
    >>> from bioio_ome_zarr.writers import OmeZarrWriterV3
    >>> from bioio_ome_zarr.writers import get_ome_zarr_writer_config_for_viz
    >>>
    >>> data = np.zeros((1, 1, 16, 256, 256), dtype="uint16")  # TCZYX
    >>> config = get_ome_zarr_writer_config_for_viz(data)
    >>> writer = OmeZarrWriterV3("output.zarr", **config)
    >>> writer.write_full_volume(data)
    """
    shape = data.shape
    dtype = data.dtype
    ndim = len(shape)

    default_axis_order = ["t", "c", "z", "y", "x"]
    axes_names = default_axis_order[-ndim:]

    scale_factors = tuple(2 if ax in ("y", "x") else 1 for ax in axes_names)

    chunk_size = chunk_size_from_memory_target(shape, dtype, 16 << 20)

    return {
        "shape": shape,
        "dtype": dtype,
        "axes_names": axes_names,
        "scale_factors": scale_factors,
        "chunk_size": chunk_size,
        "num_levels": 3,
        "image_name": "Image",
    }


def get_ome_zarr_writer_config_for_ml(
    data: Union[np.ndarray, da.Array],
) -> Dict[str, Any]:
    """
    Generate a default OmeZarrWriterV3 config for machine learning use.

    Parameters
    ----------
    data : Union[np.ndarray, dask.array.Array]
        Input data to inspect (e.g. shape TCZYX or similar).

    Returns
    -------
    config : Dict[str, Any]
        A config dictionary to pass into `OmeZarrWriterV3`.
        Customize fields after calling this function.

    Example
    -------
    >>> import numpy as np
    >>> from bioio_ome_zarr.writers import OmeZarrWriterV3
    >>> from bioio_ome_zarr.writers import get_ome_zarr_writer_config_for_ml
    >>>
    >>> data = np.zeros((1, 1, 16, 256, 256), dtype="uint16")  # TCZYX
    >>> config = get_ome_zarr_writer_config_for_ml(data)
    >>> writer = OmeZarrWriterV3("output.zarr", **config)
    >>> writer.write_full_volume(data)
    """
    shape = data.shape
    dtype = data.dtype
    ndim = len(shape)

    default_axis_order = ["t", "c", "z", "y", "x"]
    axes_names = default_axis_order[-ndim:]

    scale_factors = tuple(2 if ax in ("y", "x") else 1 for ax in axes_names)

    chunk_size = chunk_size_from_memory_target(shape, dtype, 16 << 20)

    return {
        "shape": shape,
        "dtype": dtype,
        "axes_names": axes_names,
        "scale_factors": scale_factors,
        "chunk_size": chunk_size,
        "num_levels": 3,
        "image_name": "Image",
    }
