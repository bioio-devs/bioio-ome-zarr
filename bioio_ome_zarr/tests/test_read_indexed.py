from typing import Any, Dict, List

import numpy as np
import pytest

from bioio_ome_zarr import Reader

from .conftest import LOCAL_RESOURCES_DIR

# (store, resolution level): Zarr v2, Zarr v3, zipped (.ozx), sharded Zarr v3
STORES = [
    ("s1_t7_c4_z3_Image_0.zarr", 3),
    ("s1_t7_c4_z3_Image_0_V3.zarr", 2),
    ("s1_t2_c2_z4_5d.ozx", 1),
    ("provenance_nd2_plate.ome.zarr", 0),
]


def orthogonal_reference(full: np.ndarray, dim_specs: List[Any]) -> np.ndarray:
    """Apply each spec to its own axis of the full array, independently."""
    result = full
    # Last axis first, so dropping an axis never shifts the ones still to index
    for axis in reversed(range(len(dim_specs))):
        indices = np.arange(full.shape[axis])[dim_specs[axis]]
        result = np.take(result, indices, axis=axis)
    return result


@pytest.mark.parametrize("filename, resolution_level", STORES)
@pytest.mark.parametrize(
    "dim_specs",
    [
        # Fixed indices, including a numpy int and a negative one, drop their dims
        [np.int64(1), -1, slice(None), slice(None), slice(None)],
        # Slices with steps
        [0, 0, slice(0, 3, 2), slice(1, 15, 3), slice(2, 16, 5)],
        # Negative steps, which Zarr can't slice directly
        [slice(None), 0, slice(None, None, -1), slice(9, 0, -2), slice(None)],
        # Unsorted list with a duplicate
        [0, [1, 0, 1], slice(None), slice(None), slice(None)],
        # List with a negative index
        [0, 0, [-1, 0], slice(None), slice(None)],
        # Empty selection
        [0, slice(1, 1), slice(None), slice(None), slice(None)],
        # List separated from fixed indices by a slice
        [0, 0, slice(None), [5, 1, 3], slice(None)],
        # Two lists
        [[1, 0], 0, slice(None), [5, 1, 3], slice(0, 8)],
    ],
)
def test_read_indexed_matches_orthogonal_reference(
    filename: str, resolution_level: int, dim_specs: List[Any]
) -> None:
    reader = Reader(LOCAL_RESOURCES_DIR / filename)
    reader.set_resolution_level(resolution_level)
    expected = orthogonal_reference(reader.data, dim_specs)

    actual = reader._read_indexed(reader.dims.order, dim_specs)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("filename, resolution_level", STORES)
@pytest.mark.parametrize(
    "dimension_order_out, selection",
    [
        ("ZYX", {"T": 1, "C": 0, "Y": slice(4, 12), "X": slice(0, 16)}),
        ("CYX", {"T": 0, "Z": 1, "C": [1, 0], "X": slice(2, 10, 3)}),
        # The base implementation moves the list axis to the front for these two
        ("ZYX", {"T": 0, "C": 0, "X": (0, -1)}),
        ("ZYX", {"T": 0, "C": 0, "Y": [5, 1, 3]}),
    ],
)
def test_get_image_data_reads_only_the_region(
    filename: str,
    resolution_level: int,
    dimension_order_out: str,
    selection: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = Reader(LOCAL_RESOURCES_DIR / filename)
    reader.set_resolution_level(resolution_level)
    expected = reader.get_image_dask_data(dimension_order_out, **selection).compute()

    def read_whole_level(self: Reader) -> None:
        raise AssertionError("the whole resolution level was read")

    monkeypatch.setattr(Reader, "_read_immediate", read_whole_level)
    actual = reader.get_image_data(dimension_order_out, **selection)

    np.testing.assert_array_equal(actual, expected)
