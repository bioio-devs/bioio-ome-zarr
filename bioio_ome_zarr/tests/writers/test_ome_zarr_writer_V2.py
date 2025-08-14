import pathlib
from typing import List, Optional, Tuple

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import from_ngff_zarr
from ngff_zarr.validate import validate

from bioio_ome_zarr.reader import Reader
from bioio_ome_zarr.writers import DimTuple, OmeZarrWriterV3


@pytest.mark.parametrize(
    "shape, axes_names, scale, expected_shapes, literal_level1",
    [
        (
            (4, 2, 2, 64, 32),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5), (1, 1, 1, 0.25, 0.25)),
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
            None,
        ),
        (
            (4, 2, 2, 8, 6),
            ["t", "c", "z", "y", "x"],
            None,
            [(4, 2, 2, 8, 6)],
            None,
        ),
        (
            (1, 1, 1, 13, 23),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5),),
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11)],
            None,
        ),
        (
            (1, 1, 1, 4, 4),
            ["t", "c", "z", "y", "x"],
            ((1, 1, 1, 0.5, 0.5),),
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2)],
            np.array([[5, 7], [13, 15]], dtype=np.uint16),
        ),
        (
            (4, 4),
            None,
            ((0.5, 0.5), (0.25, 0.25)),
            [(4, 4), (2, 2), (1, 1)],
            np.array([[5, 7], [13, 15]], dtype=np.uint8),
        ),
        (
            (4, 8, 8),
            None,
            ((1, 0.5, 0.5), (1, 0.25, 0.25), (1, 0.125, 0.125)),
            [(4, 8, 8), (4, 4, 4), (4, 2, 2), (4, 1, 1)],
            None,
        ),
        (
            (2, 4, 8, 8),
            ["c", "z", "y", "x"],
            ((1, 0.5, 0.5, 0.5),),
            [(2, 4, 8, 8), (2, 2, 4, 4)],
            None,
        ),
        (
            (3, 4, 8, 8),
            ["t", "z", "y", "x"],
            ((1, 0.5, 0.5, 0.5), (1, 0.25, 0.25, 0.25)),
            [(3, 4, 8, 8), (3, 2, 4, 4), (3, 1, 2, 2)],
            None,
        ),
        (
            (4, 1, 3, 8, 8),
            ["t", "c", "z", "y", "x"],
            ((0.5, 1, 0.5, 0.5, 0.5),),
            [(4, 1, 3, 8, 8), (2, 1, 1, 4, 4)],
            None,
        ),
    ],
)
def test_write_ome_zarr(
    tmp_path: pathlib.Path,
    shape: DimTuple,
    axes_names: Optional[List[str]],
    scale: Optional[Tuple[Tuple[float, ...], ...]],
    expected_shapes: List[DimTuple],
    literal_level1: Optional[np.ndarray],
) -> None:
    save_uri = tmp_path / "e.zarr"
    # Use small dtype for tiny arrays so literals remain readable
    tiny_threshold = (1 * 1 * 1 * 4 * 4) if len(shape) == 5 else (4 * 4)
    dtype = np.uint16 if int(np.prod(shape)) <= tiny_threshold else np.uint8

    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    writer_kwargs = dict(
        store=str(save_uri),
        shape=shape,
        dtype=data.dtype,
        zarr_format=2,
        scale=scale,
        image_name="TEST",
    )
    if axes_names:
        writer_kwargs["axes_names"] = axes_names

    writer = OmeZarrWriterV3(**writer_kwargs)
    writer.write_full_volume(data)

    # Shapes of each level
    ms = from_ngff_zarr(str(save_uri), validate=False, version="0.4")
    assert len(ms.images) == len(expected_shapes)
    for level, shape_expected in enumerate(expected_shapes):
        assert ms.images[level].data.shape == shape_expected

    # Optional literal content check (only for tiny hand-written cases)
    if literal_level1 is not None and len(expected_shapes) >= 2:
        grp = zarr.open(str(save_uri), mode="r")
        a1 = grp["1"][:]
        np.testing.assert_array_equal(np.squeeze(a1), literal_level1)

    # Metadata checks
    grp = zarr.open(str(save_uri), mode="r")
    attrs = grp.attrs.asdict()
    assert "multiscales" in attrs and "omero" in attrs
    validate(attrs, version="0.4", model="image", strict=False)
    ms_attr = attrs["multiscales"][0]
    assert len(ms_attr["datasets"]) == len(expected_shapes)


@pytest.mark.parametrize(
    "writer_axes, writer_shape, src_axes, src_shape, scale, "
    "expect_level1_literal, expect_error",
    [
        (
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ((1, 1, 0.5, 0.5),),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        (
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ((1, 1, 1, 0.5, 0.5),),
            None,
            None,
        ),
        (
            ["t", "c", "y", "x"],
            (2, 1, 8, 8),
            ["t", "y", "x"],
            (2, 8, 8),
            ((1, 1, 0.5, 0.5),),
            None,
            ValueError,
        ),
    ],
)
def test_write_timepoints_v2_array_only(
    tmp_path: pathlib.Path,
    writer_axes: List[str],
    writer_shape: Tuple[int, ...],
    src_axes: List[str],
    src_shape: Tuple[int, ...],
    scale: Optional[Tuple[Tuple[float, ...], ...]],
    expect_level1_literal: Optional[np.ndarray],
    expect_error: Optional[type],
) -> None:
    src = np.arange(np.prod(src_shape), dtype=np.uint16).reshape(src_shape)

    out_store = tmp_path / "out_array.zarr"
    writer = OmeZarrWriterV3(
        store=str(out_store),
        shape=writer_shape,
        dtype=src.dtype,
        zarr_format=2,
        axes_names=writer_axes,
        scale=scale,
        image_name="TEST",
    )

    arr = da.from_array(
        src,
        chunks=tuple(max(1, s // 2) for s in src.shape),
    )

    def run_write_timepoints() -> None:
        writer.write_timepoints(arr, tbatch=2)

    if expect_error:
        with pytest.raises(expect_error):
            run_write_timepoints()
        return

    run_write_timepoints()

    root = zarr.open_group(str(out_store), mode="r")
    np.testing.assert_array_equal(root["0"][:], src)
    if expect_level1_literal is not None:
        np.testing.assert_array_equal(root["1"][:], expect_level1_literal)


@pytest.mark.parametrize(
    "writer_axes, writer_shape, src_axes, src_shape, scale, "
    "expect_level1_literal, expect_error",
    [
        (
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ["t", "c", "y", "x"],
            (1, 1, 4, 4),
            ((1, 1, 0.5, 0.5),),
            np.array([[[[5, 7], [13, 15]]]], dtype=np.uint16),
            None,
        ),
        (
            ["t", "y", "x"],
            (5, 8, 8),
            ["t", "y", "x"],
            (5, 8, 8),
            ((1, 0.5, 0.5),),
            None,
            None,
        ),
        (
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ["t", "c", "z", "y", "x"],
            (2, 2, 4, 16, 16),
            ((1, 1, 1, 0.5, 0.5),),
            None,
            None,
        ),
        (
            ["t", "c", "y", "x"],
            (2, 1, 8, 8),
            ["t", "y", "x"],
            (2, 8, 8),
            ((1, 1, 0.5, 0.5),),
            None,
            ValueError,
        ),
    ],
)
def test_write_timepoints_v2_reader_only(
    tmp_path: pathlib.Path,
    writer_axes: List[str],
    writer_shape: Tuple[int, ...],
    src_axes: List[str],
    src_shape: Tuple[int, ...],
    scale: Optional[Tuple[Tuple[float, ...], ...]],
    expect_level1_literal: Optional[np.ndarray],
    expect_error: Optional[type],
) -> None:
    src = np.arange(np.prod(src_shape), dtype=np.uint16).reshape(src_shape)

    in_store = tmp_path / "in_reader.zarr"
    in_root = zarr.open_group(str(in_store), mode="w")
    in_root.create_dataset(
        "0",
        shape=src.shape,
        dtype=src.dtype,
        chunks=tuple(max(1, s // 2) for s in src.shape),
    )[:] = src
    in_root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a} for a in src_axes],
            "datasets": [{"path": "0"}],
        }
    ]

    reader = Reader(str(in_store))

    out_store = tmp_path / "out_reader.zarr"
    writer = OmeZarrWriterV3(
        store=str(out_store),
        shape=writer_shape,
        dtype=src.dtype,
        zarr_format=2,
        axes_names=writer_axes,
        scale=scale,
        image_name="TEST",
    )

    def run_write_timepoints() -> None:
        writer.write_timepoints(reader, tbatch=2)

    if expect_error:
        with pytest.raises(expect_error):
            run_write_timepoints()
        return

    run_write_timepoints()

    root = zarr.open_group(str(out_store), mode="r")
    np.testing.assert_array_equal(root["0"][:], src)
    if expect_level1_literal is not None:
        np.testing.assert_array_equal(root["1"][:], expect_level1_literal)
