#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr
import zarr
from bioio_base import constants, dimensions, exceptions, io, reader, types
from fsspec.spec import AbstractFileSystem
from s3fs import S3FileSystem
from zarr.core.group import GroupMetadata

from . import utils as metadata_utils

STORAGE_OPTIONS = {"anon": True}


class Reader(reader.Reader):
    """
    Unified Zarr Reader that supports both V2 and V3 metadata schemas.

    On initialization, this reader expands the provided image path using
    io.pathlike_to_fs (which returns _fs and _path) and then opens the Zarr group.

    It distinguishes between the V3 schema (if "ome" key exists) and the V2 schema
    (if not) by setting self._version accordingly, and then uses that flag to drive
    metadata extraction for scenes, dims, scale arrays, etc.

    Attributes:
      _fs: An AbstractFileSystem instance (from fsspec).
      _path: The resolved image path.
      _zarr: The opened Zarr group.
      _version: 3 for V3 metadata (if "ome" key exists), otherwise 2.
    """

    _xarray_dask_data: Optional[xr.DataArray] = None
    _xarray_data: Optional[xr.DataArray] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    _current_resolution_level: int = 0

    _channel_names: Optional[List[str]] = None

    _fs: AbstractFileSystem  # now declared from io.pathlike_to_fs output
    _path: str
    _zarr: zarr.Group
    _version: int  # 3 for V3, 2 for V2

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
    ):
        # Expand details of provided image.
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=False,
            fs_kwargs=fs_kwargs,
        )
        if isinstance(self._fs, S3FileSystem):
            self._path = str(image)

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                "Could not find a .zgroup or .zarray file at the provided path.",
            )
        try:
            if isinstance(self._fs, S3FileSystem):
                self._zarr = zarr.open_group(
                    self._path, mode="r", storage_options={"anon": True}
                )
            else:
                self._zarr = zarr.open_group(self._path, mode="r")
        except Exception as e:
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                f"Failed to open the Zarr group: {e}",
            )
        # Determine metadata schema version: if "ome" key exists, assume V3; else V2.
        self._version = 3 if "ome" in self._zarr.attrs else 2

    @staticmethod
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            # If the filesystem indicates S3, pass storage_options accordingly.
            if isinstance(fs, S3FileSystem):
                zarr.open_group(path, mode="r", storage_options={"anon": True})
            else:
                zarr.open_group(path, mode="r")
            return True
        except Exception:
            return False

    @classmethod
    def is_supported_image(
        cls, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}, **kwargs: Any
    ) -> bool:
        if isinstance(image, (str, Path)):
            return cls._is_supported_image(None, str(Path(image).resolve()), **kwargs)
        else:
            return reader.Reader.is_supported_image(
                cls, image, fs_kwargs=fs_kwargs, **kwargs
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        key = "multiscales"
        if self._version == 3:
            scenes_meta = self._zarr.attrs.get("ome", {}).get(key, [])
        else:
            scenes_meta = self._zarr.attrs.get(key, [])
        if not scenes_meta:
            raise ValueError("No multiscales metadata found.")
        if all("name" in scene for scene in scenes_meta) and (
            len({scene["name"] for scene in scenes_meta}) == len(scenes_meta)
        ):
            return tuple(scene["name"] for scene in scenes_meta)
        return tuple(
            metadata_utils.generate_ome_image_id(i) for i in range(len(scenes_meta))
        )

    @property
    def ome_dims(self) -> Tuple[str, ...]:
        key = "multiscales"
        if self._version == 3:
            ms = self._zarr.attrs.get("ome", {}).get(key, [])[self._current_scene_index]
        else:
            ms = self._zarr.attrs.get(key, [])[self._current_scene_index]
        axes = ms.get("axes", [])
        if axes:
            return tuple(ax["name"].upper() for ax in axes)
        else:
            # No axes metadata, use shape of the first dataset array to guess dims.
            datasets = ms.get("datasets", [])
            if datasets:
                data_path = datasets[0].get("path")
                if data_path is not None:
                    arr = self._zarr[data_path]
                    return tuple(Reader._guess_dim_order(arr.shape))
            # If no dataset exists, return an empty tuple.
            return tuple()

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        key = "multiscales"
        if self._version == 3:
            ms = self._zarr.attrs.get("ome", {}).get(key, [])[self._current_scene_index]
        else:
            ms = self._zarr.attrs.get(key, [])[self._current_scene_index]
        datasets = ms.get("datasets", [])
        return tuple(range(len(datasets)))

    @property
    def current_scene(self) -> str:
        return self.scenes[self._current_scene_index]

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(delayed=False)

    def _xarr_format(self, delayed: bool) -> xr.DataArray:
        key = "multiscales"
        if self._version == 3:
            ms = self._zarr.attrs.get("ome", {}).get(key, [])[self._current_scene_index]
        else:
            ms = self._zarr.attrs.get(key, [])[self._current_scene_index]
        try:
            dataset = ms.get("datasets", [])[self._current_resolution_level]
        except IndexError:
            raise ValueError("Invalid resolution level index.")
        data_path = dataset.get("path")
        if data_path is None:
            raise ValueError("Dataset metadata is missing a 'path'.")
        arr = self._zarr[data_path]
        if not delayed:
            data = arr[:]
        else:
            try:
                import dask.array as da

                data = da.from_array(arr, chunks=arr.chunks)
            except ImportError:
                data = arr
        dims = self.ome_dims
        coords = self._get_coords(
            list(dims),
            data.shape,
            scene=self.current_scene,
            channel_names=self.channel_names,
        )
        return xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            attrs={constants.METADATA_UNPROCESSED: self._zarr.attrs},
        )

    @staticmethod
    def _get_coords(
        dims: List[str],
        shape: Tuple[int, ...],
        scene: str,
        channel_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        coords: Dict[str, Any] = {}
        if dimensions.DimensionNames.Channel in dims:
            if channel_names is None:
                coords[dimensions.DimensionNames.Channel] = [
                    metadata_utils.generate_ome_channel_id(image_id=scene, channel_id=i)
                    for i in range(shape[dims.index(dimensions.DimensionNames.Channel)])
                ]
            else:
                coords[dimensions.DimensionNames.Channel] = channel_names
        return coords

    def _get_scale_array(self, dims: Tuple[str, ...]) -> List[float]:
        """
        Retrieves the effective scale array for the current dataset.

        It first checks if the multiscale metadata (at key "multiscales") contains
        an overall coordinateTransformation (e.g. a universal scaling factor) and
        uses that as a multiplier. Then it retrieves the dataset's
        coordinateTransformations scale array and multiplies each corresponding
        element.

        Assumes the final scale array is provided in the same order as dims.
        """
        key = "multiscales"
        if self._version == 3:
            ms = self._zarr.attrs.get("ome", {}).get(key, [])[self._current_scene_index]
        else:
            ms = self._zarr.attrs.get(key, [])[self._current_scene_index]

        # Get overall (universal) scale from the multiscale level if present.
        overall_ct = ms.get("coordinateTransformations", [])
        if overall_ct:
            overall_scale = overall_ct[0].get("scale", [1.0] * len(dims))
        else:
            overall_scale = [1.0] * len(dims)

        datasets = ms.get("datasets", [])
        if self._current_resolution_level >= len(datasets):
            raise ValueError("Invalid resolution level index.")
        dataset = datasets[self._current_resolution_level]
        try:
            ds_ct = dataset["coordinateTransformations"][0]
        except (KeyError, IndexError):
            raise ValueError("Missing coordinateTransformations in dataset metadata.")
        ds_scale = ds_ct.get("scale")
        if ds_scale is None or len(ds_scale) != len(dims):
            raise ValueError(
                "Dataset scale missing or does not match the number of dimensions."
            )

        # Compute effective scales: element-wise multiplication.
        effective_scale = [overall_scale[i] * ds_scale[i] for i in range(len(dims))]
        return effective_scale

    @property
    def time_interval(self) -> Optional[float]:
        dims = self.ome_dims
        if "T" not in dims:
            return None
        scale_array = self._get_scale_array(dims)
        return scale_array[dims.index("T")]

    def _get_pixel_size(
        self, dims: Tuple[str, ...]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        scale_array = self._get_scale_array(dims)
        z = scale_array[dims.index("Z")] if "Z" in dims else None
        y = scale_array[dims.index("Y")] if "Y" in dims else None
        x = scale_array[dims.index("X")] if "X" in dims else None
        return (z, y, x)

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        try:
            dims = self.ome_dims
            z, y, x = self._get_pixel_size(dims)
        except Exception as e:
            warnings.warn(f"Could not parse physical pixel sizes: {e}")
            z, y, x = None, None, None
        return types.PhysicalPixelSizes(z, y, x)

    @property
    def channel_names(self) -> Optional[List[str]]:
        if self._channel_names is None:
            if self._version == 3:
                channels = (
                    self._zarr.attrs.get("ome", {}).get("omero", {}).get("channels", [])
                )
            else:
                channels = self._zarr.attrs.get("omero", {}).get("channels", [])
            try:
                self._channel_names = [str(ch["label"]) for ch in channels]
            except Exception:
                self._channel_names = None
        return self._channel_names

    @property
    def metadata(self) -> GroupMetadata:
        """
        Returns the metadata of the underlying Zarr group.
        Expected to be of type GroupMetadata.
        """
        return self._zarr.metadata

    @property
    def scale(self) -> types.Scale:
        """
        Constructs and returns a Scale object where:
          - T is taken from the property time_interval,
          - Z, Y, and X are taken from physical_pixel_sizes,
          - and C (channel) is taken from the dataset's scale array if present
        """
        dims = self.ome_dims
        scale_array = self._get_scale_array(dims)
        c = scale_array[dims.index("C")] if "C" in dims else None
        return types.Scale(
            T=self.time_interval,
            C=c,
            Z=self.physical_pixel_sizes.Z,
            Y=self.physical_pixel_sizes.Y,
            X=self.physical_pixel_sizes.X,
        )
