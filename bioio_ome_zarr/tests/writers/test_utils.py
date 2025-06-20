from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
from dask import array as da

from bioio_ome_zarr.writers import (
    chunk_size_from_memory_target,
    compute_level_chunk_sizes_zslice,
    compute_level_shapes,
    get_scale_ratio,
    resize,
)


@pytest.mark.parametrize(
    "shape, dtype, target, expected",
    [
        # original 5D uint16 cases
        ((1, 1, 1, 128, 128), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((1, 1, 1, 127, 127), np.uint16, 1024, (1, 1, 1, 15, 15)),
        ((1, 1, 1, 129, 129), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((7, 11, 128, 128, 128), np.uint16, 1024, (1, 1, 8, 8, 8)),
        # 2D uint8 (YX) with 1 KiB target
        ((256, 256), np.uint8, 1024, (32, 32)),
        # 3D uint8 (ZYX) with ~1 KiB target
        ((10, 20, 30), np.uint8, 1000, (5, 10, 15)),
        # 4D uint8 (CZYX) with 4 KiB target
        ((2, 4, 64, 64), np.uint8, 4096, (1, 2, 32, 32)),
        # 5D float32 (TCZYX) with 256 KiB target
        ((1, 1, 64, 64, 64), np.float32, 256 * 1024, (1, 1, 32, 32, 32)),
        # >5D without explicit order should xfail
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            np.uint8,
            1024,
            None,
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="Shapes >5D without `order` must raise",
            ),
        ),
    ],
)
def test_chunk_size_from_memory_target(
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    target: int,
    expected: Optional[Tuple[int, ...]],
) -> None:
    """
    Parameterized test for chunk_size_from_memory_target:
      - Valid 2D–5D cases (various dtypes & sizes)
      - >5D case xfails with ValueError when order=None
    """
    out = chunk_size_from_memory_target(shape, dtype, target)
    assert out == expected


@pytest.mark.parametrize(
    "base_shape, axis_names, axis_factors, max_levels, expected",
    [
        ((64, 64), ["y", "x"], (2, 2), 3, [(64, 64), (32, 32), (16, 16)]),
        (
            (8, 64, 64),
            ["z", "y", "x"],
            (1, 2, 2),
            4,
            [(8, 64, 64), (8, 32, 32), (8, 16, 16), (8, 8, 8)],
        ),
        (
            (5, 32, 64, 64),
            ["t", "z", "y", "x"],
            (1, 1, 2, 2),
            2,
            [(5, 32, 64, 64), (5, 32, 32, 32)],
        ),
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            (1, 1, 1, 2, 2),
            2,
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
        ),
    ],
)
def test_compute_level_shapes_v3(
    base_shape: Tuple[int, ...],
    axis_names: List[str],
    axis_factors: Tuple[int, ...],
    max_levels: int,
    expected: List[Tuple[int, ...]],
) -> None:
    """
    Test the V3 compute_level_shapes signature.
    """
    out = compute_level_shapes(base_shape, axis_names, axis_factors, max_levels)
    assert out == expected


def test_compute_level_shapes_legacy() -> None:
    """
    Test the legacy compute_level_shapes signature.
    """
    base_shape: Tuple[int, ...] = (1, 1, 1, 4, 4)
    scaling: Tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 2.0)
    levels: int = 2
    expected: List[Tuple[int, ...]] = [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)]
    out = compute_level_shapes(base_shape, scaling, levels)
    assert out == expected


def test_resize_simple() -> None:
    """
    Test the resize utility for a small 2D case.
    """
    d = da.from_array(np.arange(16).reshape(4, 4), chunks=(2, 2))
    out = resize(d, (2, 2))
    assert out.shape == (2, 2)
    assert out.dtype == d.dtype


@pytest.mark.parametrize(
    "in_shapes, expected",
    [
        (
            [
                (512, 4, 100, 1000, 1000),
                (512, 4, 100, 500, 500),
                (512, 4, 100, 250, 250),
            ],
            [(1, 1, 1, 1000, 1000), (1, 1, 4, 500, 500), (1, 1, 16, 250, 250)],
        ),
    ],
)
def test_compute_level_chunk_sizes_zslice(
    in_shapes: List[Tuple[int, ...]],
    expected: List[Tuple[int, ...]],
) -> None:
    """
    Test compute_level_chunk_sizes_zslice utility.
    """
    out = compute_level_chunk_sizes_zslice(in_shapes)
    assert out == expected


def test_get_scale_ratio() -> None:
    """
    Test get_scale_ratio utility.
    """
    lvl0: Tuple[int, ...] = (4, 8, 16)
    lvl1: Tuple[int, ...] = (2, 4, 8)
    assert get_scale_ratio(lvl0, lvl1) == (2.0, 2.0, 2.0)
