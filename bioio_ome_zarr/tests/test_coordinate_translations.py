from pathlib import Path
from typing import List, Literal

import numpy as np
import pytest
import zarr

from bioio_ome_zarr import Reader
from bioio_ome_zarr.writers import OMEZarrWriter


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_reader_composes_dataset_and_multiscale_translations(
    tmp_path: Path, zarr_format: Literal[2, 3]
) -> None:
    path = tmp_path / "translated.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    axes = [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    datasets = []
    for level in (0, 1):
        size = 2 * (2 - level)
        group.create_array(str(level), shape=(2, 2, size, size), dtype="uint8")
        translation = {
            "type": "translation",
            "translation": [3 + level, 0, 2 + 2 * level, -10 - 2 * level],
        }
        if zarr_format == 3 and level == 1:
            group.create_array("origin", data=np.array(translation["translation"]))
            translation = {"type": "translation", "path": "origin"}
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1, 1, 0.5 * (level + 1), 2 * (level + 1)],
                    },
                    translation,
                ],
            }
        )
    multiscale = {
        "name": "sample",
        "axes": axes,
        "datasets": datasets,
        "coordinateTransformations": [
            {"type": "scale", "scale": [2, 1, 10, 0.5]},
            {"type": "translation", "translation": [7, 0, 100, -100]},
        ],
    }
    omero = {"channels": [{"label": "GFP"}, {"label": "DAPI"}]}
    if zarr_format == 2:
        group.attrs.update(
            {"multiscales": [{**multiscale, "version": "0.4"}], "omero": omero}
        )
    else:
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [multiscale],
            "omero": omero,
        }

    reader = Reader(path)
    for level, time, y, x in [
        (0, [13, 15], [120, 125, 130, 135], [-105, -104, -103, -102]),
        (1, [15, 17], [140, 150], [-106, -104]),
    ]:
        reader.set_resolution_level(level)
        for image in (reader.xarray_data, reader.xarray_dask_data):
            np.testing.assert_array_equal(image.coords["T"], time)
            np.testing.assert_array_equal(image.coords["Y"], y)
            np.testing.assert_array_equal(image.coords["X"], x)
            np.testing.assert_array_equal(image.coords["C"], ["GFP", "DAPI"])
            assert image.coords["X"].attrs["units"] == "nanometer"
        assert reader.time_interval == 2
        assert reader.scale.Y == 5 * (level + 1)
        assert reader.scale.X == level + 1


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_writer_round_trips_translations(
    tmp_path: Path, zarr_format: Literal[2, 3]
) -> None:
    path = tmp_path / "translated.zarr"
    translations = [[1.25, -5], [1.5, -4.5]]
    writer = OMEZarrWriter(
        str(path),
        [(4, 4), (2, 2)],
        "uint8",
        zarr_format=zarr_format,
        axes_units=["micrometer", "micrometer"],
        physical_pixel_size=[0.5, 1.0],
        root_transform={"type": "translation", "translation": [2, -10]},
        translations=translations,
    )
    preview = writer.preview_metadata()
    group_metadata = preview if zarr_format == 2 else preview["ome"]
    transforms = group_metadata["multiscales"][0]["coordinateTransformations"]
    assert transforms == [
        {"type": "scale", "scale": [1.0, 1.0]},
        {"type": "translation", "translation": [2, -10]},
    ]
    for level, dataset in enumerate(group_metadata["multiscales"][0]["datasets"]):
        scale, translation = dataset["coordinateTransformations"]
        assert scale["type"] == "scale"
        assert translation == {
            "type": "translation",
            "translation": translations[level],
        }
    assert not path.exists()

    writer.write_full_volume(np.arange(16, dtype=np.uint8).reshape(4, 4))
    reader = Reader(path)
    np.testing.assert_array_equal(
        reader.xarray_data.coords["Y"], [3.25, 3.75, 4.25, 4.75]
    )
    reader.set_resolution_level(1)
    np.testing.assert_array_equal(reader.xarray_data.coords["Y"], [3.5, 4.5])
    np.testing.assert_array_equal(reader.xarray_data.coords["X"], [-14.5, -12.5])


@pytest.mark.parametrize(
    "translations",
    [[[1, 2]], [[0, 0], [1]], [[0, 0], [1, np.nan]]],
)
def test_invalid_writer_translations(
    tmp_path: Path, translations: List[List[float]]
) -> None:
    path = tmp_path / "translated.zarr"
    with pytest.raises(ValueError, match="translations must provide one finite offset"):
        OMEZarrWriter(str(path), [(4, 4), (2, 2)], "uint8", translations=translations)
    assert not path.exists()


def test_reader_keeps_image_and_scale_if_translation_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "translated.zarr"
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    writer = OMEZarrWriter(str(path), image.shape, image.dtype)
    writer.write_full_volume(image)
    assert "Y" not in Reader(path).xarray_data.coords

    group = zarr.open_group(path, mode="a")
    ome = group.attrs["ome"]
    ome["multiscales"][0]["datasets"][0]["coordinateTransformations"].append(
        {"type": "translation", "translation": [1]}
    )
    group.attrs["ome"] = ome

    reader = Reader(path)
    with pytest.warns(UserWarning, match="Could not parse coordinate translations"):
        np.testing.assert_array_equal(reader.xarray_data, image)
    assert reader.scale.Y == reader.scale.X == 1
