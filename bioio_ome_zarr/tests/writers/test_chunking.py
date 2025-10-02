import numpy as np
import pytest

from ...writers.utils import (
    DimSeq,
    PerLevelDimSeq,
    multiscale_chunk_size_from_memory_target,
)


def _nbytes(shape: DimSeq, dtype: str = "uint16") -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


@pytest.mark.parametrize(
    "level_shapes,dtype,mem_target,expected",
    [
        pytest.param(  # 2D YX
            [(64, 64), (32, 32), (16, 16)],
            "uint16",
            16 << 20,
            [(64, 64), (32, 32), (16, 16)],
            id="passthrough_2d",
        ),
        pytest.param(  # 5D TCZYX
            [(2, 3, 4, 64, 64), (2, 3, 4, 32, 32), (2, 3, 4, 16, 16)],
            "uint16",
            16 << 20,
            [(2, 3, 4, 64, 64), (2, 3, 4, 32, 32), (2, 3, 4, 16, 16)],
            id="passthrough_5d",
        ),
    ],
)
def test_passthrough(
    level_shapes: PerLevelDimSeq, dtype: str, mem_target: int, expected: PerLevelDimSeq
) -> None:
    chunks = multiscale_chunk_size_from_memory_target(level_shapes, dtype, mem_target)
    assert chunks == expected


@pytest.mark.parametrize(
    "level_shapes,expected",
    [
        pytest.param(
            [(100, 100), (37, 37), (19, 19)],
            [(100, 100), (37, 37), (19, 19)],
            id="irregular_downsample",
        ),
    ],
)
def test_irregular_downsampling(
    level_shapes: PerLevelDimSeq, expected: PerLevelDimSeq
) -> None:
    chunks = multiscale_chunk_size_from_memory_target(level_shapes, "uint16", 16 << 20)
    assert chunks == expected


@pytest.mark.parametrize(
    "level_shapes,dtype,mem_target,expected",
    [
        pytest.param(
            [(2, 3, 4, 64, 64), (2, 3, 4, 32, 32), (2, 3, 4, 16, 16)],
            "uint16",
            8 << 10,  # 8 KiB = bytes((1,1,1,64,64))
            [
                (1, 1, 1, 64, 64),
                (1, 1, 4, 32, 32),
                (1, 3, 4, 16, 16),
            ],
            id="budget_progression_5d_8KiB_Z_only",
        ),
        pytest.param(
            [(2, 3, 4, 64, 64), (2, 3, 4, 32, 32), (2, 3, 4, 16, 16)],
            "uint16",
            32 << 10,  # 32 KiB = bytes((1,1,4,64,64)) → Z full
            [
                (1, 1, 4, 64, 64),
                (1, 3, 4, 32, 32),
                (2, 3, 4, 16, 16),
            ],
            id="budget_progression_5d_32KiB_Z_full_then_C_then_T",
        ),
        pytest.param(
            [(2, 3, 4, 64, 64), (2, 3, 4, 32, 32), (2, 3, 4, 16, 16)],
            "uint16",
            64 << 10,  # 64 KiB → C grows at L0, T grows at L1
            [
                (1, 2, 4, 64, 64),
                (2, 3, 4, 32, 32),
                (2, 3, 4, 16, 16),
            ],
            id="budget_progression_5d_64KiB_C_growth",
        ),
        pytest.param(
            [(2, 3, 4, 64, 64), (2, 3, 4, 32, 32), (2, 3, 4, 16, 16)],
            "uint16",
            192 << 10,  # 192 KiB ≈ full
            [
                (2, 3, 4, 64, 64),
                (2, 3, 4, 32, 32),
                (2, 3, 4, 16, 16),
            ],
            id="budget_progression_5d_full",
        ),
    ],
)
def test_budget_progression(
    level_shapes: PerLevelDimSeq, dtype: str, mem_target: int, expected: PerLevelDimSeq
) -> None:
    chunks = multiscale_chunk_size_from_memory_target(level_shapes, dtype, mem_target)
    assert chunks == expected
    for ch in chunks:
        assert _nbytes(ch, dtype) <= mem_target


@pytest.mark.parametrize(
    "level_shapes,dtype,memory_target,expected",
    [
        # compact 2–3 level coverage
        pytest.param(
            [(1024, 1024), (512, 512), (256, 256)],
            "uint16",
            1 << 20,  # 1 MiB
            [(512, 1024), (512, 512), (256, 256)],
            id="small_2d_yx",
        ),
        pytest.param(
            [(32, 1024, 1024), (32, 512, 512)],
            "float32",
            4 << 20,  # 4 MiB
            [(1, 1024, 1024), (4, 512, 512)],
            id="small_3d_zyx",
        ),
        pytest.param(
            [(3, 32, 1024, 1024), (3, 32, 512, 512)],
            "uint16",
            8 << 20,  # 8 MiB
            [(1, 4, 1024, 1024), (1, 16, 512, 512)],
            id="small_4d_czyx",
        ),
        pytest.param(
            [(10, 2, 32, 1024, 1024), (10, 2, 32, 512, 512)],
            "uint16",
            16 << 20,  # 16 MiB
            [(1, 1, 8, 1024, 1024), (1, 1, 32, 512, 512)],
            id="small_5d_tczyx",
        ),
        # very large shapes, 5 levels
        pytest.param(
            [(8192, 8192), (4096, 4096), (2048, 2048), (1024, 1024), (512, 512)],
            "uint16",
            8 << 20,  # 8 MiB
            [(512, 8192), (1024, 4096), (2048, 2048), (1024, 1024), (512, 512)],
            id="large_2d_yx_5levels",
        ),
        pytest.param(
            [
                (64, 4096, 4096),
                (64, 2048, 2048),
                (64, 1024, 1024),
                (64, 512, 512),
                (64, 256, 256),
            ],
            "uint16",
            8 << 20,  # 8 MiB
            [
                (1, 1024, 4096),
                (1, 2048, 2048),
                (4, 1024, 1024),
                (16, 512, 512),
                (64, 256, 256),
            ],
            id="large_3d_zyx_5levels",
        ),
        pytest.param(
            [
                (8, 64, 4096, 4096),
                (8, 64, 2048, 2048),
                (8, 64, 1024, 1024),
                (8, 64, 512, 512),
                (8, 64, 256, 256),
            ],
            "uint16",
            16 << 20,  # 16 MiB
            [
                (1, 1, 2048, 4096),
                (1, 2, 2048, 2048),
                (1, 8, 1024, 1024),
                (1, 32, 512, 512),
                (2, 64, 256, 256),
            ],
            id="large_4d_czyx_5levels",
        ),
        pytest.param(
            [
                (16, 4, 64, 2048, 2048),
                (16, 4, 64, 1024, 1024),
                (16, 4, 64, 512, 512),
                (16, 4, 64, 256, 256),
                (16, 4, 64, 128, 128),
            ],
            "uint16",
            16 << 20,  # 16 MiB
            [
                (1, 1, 2, 2048, 2048),
                (1, 1, 8, 1024, 1024),
                (1, 1, 32, 512, 512),
                (1, 2, 64, 256, 256),
                (2, 4, 64, 128, 128),
            ],
            id="large_5d_tczyx_5levels",
        ),
    ],
)
def test_multiscale_chunk_size_from_memory_target(
    level_shapes: PerLevelDimSeq,
    dtype: str,
    memory_target: int,
    expected: PerLevelDimSeq,
) -> None:
    chunks = multiscale_chunk_size_from_memory_target(
        level_shapes, dtype, memory_target
    )
    assert chunks == expected

    for lvl, (shape, chunk) in enumerate(zip(level_shapes, chunks)):
        assert all(1 <= c <= s for c, s in zip(chunk, shape)), f"[L{lvl}] bounds"
        assert _nbytes(chunk, dtype) <= memory_target, f"[L{lvl}] bytes"
