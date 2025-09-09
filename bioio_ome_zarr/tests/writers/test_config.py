from typing import Tuple

import numpy as np
import pytest

from bioio_ome_zarr.writers import (
    get_default_config_for_ml,
    get_default_config_for_viz,
)


@pytest.mark.parametrize("shape", [(1, 1, 4, 16, 8), (4, 4)])
def test_viz_preset_cfg(shape: Tuple[int, ...]) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    cfg = get_default_config_for_viz(data)

    # Required keys
    assert set(cfg.keys()) == {"shape", "dtype", "scale", "chunk_shape"}

    # Shape and dtype match input
    assert cfg["shape"] == shape
    assert cfg["dtype"] == data.dtype

    # Scale has 2 pyramid levels (XY downsampled)
    assert isinstance(cfg["scale"], tuple)
    assert len(cfg["scale"]) == 2
    for vec in cfg["scale"]:
        assert len(vec) == len(shape)
        # Y and X entries should be < 1.0
        assert vec[-1] < 1.0 and vec[-2] < 1.0

    # Chunk shape is a tuple with ndim entries
    assert isinstance(cfg["chunk_shape"], tuple)
    assert len(cfg["chunk_shape"]) == len(shape)
    assert all(isinstance(c, int) and c > 0 for c in cfg["chunk_shape"])


@pytest.mark.parametrize("shape", [(1, 1, 8, 32, 16), (1, 32, 16)])
def test_ml_preset_cfg(shape: Tuple[int, ...]) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    cfg = get_default_config_for_ml(data)

    # Required keys
    assert set(cfg.keys()) == {"shape", "dtype", "scale", "chunk_shape"}

    # Shape and dtype match input
    assert cfg["shape"] == shape
    assert cfg["dtype"] == data.dtype

    # No pyramid
    assert cfg["scale"] == tuple()

    # Chunk shape is valid
    assert isinstance(cfg["chunk_shape"], tuple)
    assert len(cfg["chunk_shape"]) == len(shape)
