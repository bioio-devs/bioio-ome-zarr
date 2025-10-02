import inspect
from typing import List, Tuple

import numpy as np
import pytest
import zarr

from bioio_ome_zarr.writers import (
    OMEZarrWriter,
    get_default_config_for_ml,
    get_default_config_for_viz,
)


def _allowed_writer_keys() -> List[str]:
    """Kwarg names accepted by OMEZarrWriter.__init__ (minus 'self')."""
    sig = inspect.signature(OMEZarrWriter.__init__)
    return [name for name in sig.parameters if name != "self"]


@pytest.mark.parametrize("level0_shape", [(1, 1, 4, 16, 8), (4, 4)])
def test_viz_preset_cfg(level0_shape: Tuple[int, ...]) -> None:
    data = np.zeros(level0_shape, dtype=np.uint8)
    cfg = get_default_config_for_viz(data)

    # API valid
    assert set(cfg.keys()) == {"level_shapes", "dtype", "chunk_shape"}
    assert set(cfg.keys()).issubset(set(_allowed_writer_keys()))

    # Basic semantics
    assert cfg["dtype"] == data.dtype
    level_shapes = cfg["level_shapes"]
    assert len(level_shapes) == 3
    assert tuple(level_shapes[0]) == tuple(level0_shape)

    # XY pyramid: halve and quarter Y/X (floor)
    ndim = len(level0_shape)
    y_idx, x_idx = ndim - 2, ndim - 1
    expected_l1 = list(level0_shape)
    expected_l1[y_idx] = max(1, level0_shape[y_idx] // 2)
    expected_l1[x_idx] = max(1, level0_shape[x_idx] // 2)
    expected_l2 = list(level0_shape)
    expected_l2[y_idx] = max(1, level0_shape[y_idx] // 4)
    expected_l2[x_idx] = max(1, level0_shape[x_idx] // 4)
    assert tuple(level_shapes[1]) == tuple(expected_l1)
    assert tuple(level_shapes[2]) == tuple(expected_l2)

    # Chunk shape
    chunk_shape = cfg["chunk_shape"]
    assert isinstance(chunk_shape, tuple)
    assert len(chunk_shape) == ndim
    assert all(isinstance(c, int) and c > 0 for c in chunk_shape)

    # Constructor accepts the preset (no IO performed yet)
    _ = OMEZarrWriter(store=zarr.storage.MemoryStore(), **cfg)


@pytest.mark.parametrize("level0_shape", [(1, 1, 8, 32, 16), (1, 32, 16)])
def test_ml_preset_cfg(level0_shape: Tuple[int, ...]) -> None:
    data = np.zeros(level0_shape, dtype=np.uint8)
    cfg = get_default_config_for_ml(data)

    # API valid
    assert set(cfg.keys()) == {"level_shapes", "dtype", "chunk_shape"}
    assert set(cfg.keys()).issubset(set(_allowed_writer_keys()))

    # Basic semantics
    assert cfg["dtype"] == data.dtype
    assert len(cfg["level_shapes"]) == 1
    assert tuple(cfg["level_shapes"][0]) == tuple(level0_shape)

    # Chunk shape
    chunk_shape = cfg["chunk_shape"]
    assert isinstance(chunk_shape, tuple)
    assert len(chunk_shape) == len(level0_shape)
    assert all(isinstance(c, int) and c > 0 for c in chunk_shape)

    # If Z exists (… Z Y X), its chunk is 1
    if len(level0_shape) >= 3:
        z_idx = len(level0_shape) - 3
        assert chunk_shape[z_idx] == 1

    # Constructor accepts the preset (no IO performed yet)
    _ = OMEZarrWriter(store=zarr.storage.MemoryStore(), **cfg)
