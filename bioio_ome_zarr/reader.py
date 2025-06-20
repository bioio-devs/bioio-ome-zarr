#!/usr/bin/env python
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import xarray as xr
import zarr
from bioio_base import constants, dimensions, exceptions, io, reader, types
from fsspec.spec import AbstractFileSystem
from ome_types import OME
from ome_types.model import Channel, Image, Pixels, PixelType
from s3fs import S3FileSystem
from zarr.core.group import GroupMetadata

from . import utils as metadata_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Reader(reader.Reader):
    """
    The main class of the `bioio_ome_zarr` plugin. This class is subclass
    of the abstract class `reader` (`BaseReader`) in `bioio-base`.

    Parameters
    ----------
    image: types.PathLike
        String or Path to the ZARR top directory.
    fs_kwargs: Dict[str, Any]
        Passed to fsspec. For public S3 buckets, use {"anon": True}.
    """

    _channel_names: Optional[List[str]] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _zarr: zarr.Group
    _ome_metadata: OME

    _fs: AbstractFileSystem
    _path: str

    _current_scene_index: int = 0

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

        # io.pathlike_to_fs clips s3 paths
        if isinstance(self._fs, S3FileSystem):
            self._path = str(image)

        if not self._is_supported_image(self._fs, self._path, fs_kwargs=fs_kwargs):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                "Could not parse a zarr file at the provided path.",
            )

        self._zarr = zarr.open_group(self._path, mode="r", storage_options=fs_kwargs)

        self._multiscales_metadata = self._zarr.attrs.get("ome", {}).get(
            "multiscales"
        ) or self._zarr.attrs.get("multiscales", [])

        self._channel_metadata = (
            self._zarr.attrs.get("ome", {}).get("omero", {}).get("channels")
            or self._zarr.attrs.get("omero", {}).get("channels")
            or []
        )

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, fs_kwargs: Dict[str, Any], **kwargs: Any
    ) -> bool:
        # Warn users who are reading from s3 with no storage options
        if isinstance(fs, S3FileSystem) and fs_kwargs == {}:
            warnings.warn(
                "Warning: you are reading a S3 file without specifying fs_kwargs, "
                "Consider providing fs_kwargs (e.g., {{'anon': True}} for public s3)"
                "to ensure accurate reading."
            )
        try:
            zarr.open_group(path, mode="r", storage_options=fs_kwargs)
            return True
        except Exception:
            return False

    @classmethod
    def is_supported_image(
        cls, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}, **kwargs: Any
    ) -> bool:
        if isinstance(image, (str, Path)):
            return cls._is_supported_image(
                None, str(Path(image).resolve()), fs_kwargs, **kwargs
            )
        else:
            return reader.Reader.is_supported_image(
                cls, image, fs_kwargs=fs_kwargs, **kwargs
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene ids in the file.
        """
        if self._scenes is None:
            scenes = [scene.get("name") for scene in self._multiscales_metadata]

            # Check that every name exists and that they're all unique
            if all(scenes) and len(set(scenes)) == len(scenes):
                self._scenes = tuple(scenes)
            else:
                # Otherwise generate default IDs by index
                self._scenes = tuple(
                    metadata_utils.generate_ome_image_id(i)
                    for i in range(len(self._multiscales_metadata))
                )

        return self._scenes

    def _get_ome_dims(self) -> Tuple[str, ...]:
        multiscales = self._multiscales_metadata[self._current_scene_index]
        axes = multiscales.get("axes", [])
        if axes:
            return tuple(ax["name"].upper() for ax in axes)
        datasets = multiscales.get("datasets", [])
        arr = self._zarr[datasets[0]["path"]]
        return tuple(Reader._guess_dim_order(arr.shape))

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        resolution_levels: Tuple[str, ...]
            Return the available resolution levels for the current scene.
            By default these are ordered from highest resolution to lowest
            resolution.
        """
        multiscales = self._multiscales_metadata[self._current_scene_index]
        return tuple(range(len(multiscales.get("datasets", []))))

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(delayed=False)

    def _xarr_format(self, delayed: bool) -> xr.DataArray:
        """
        Build an xarray.DataArray for the current scene and resolution level.

        Parameters
        ----------
        delayed : bool
            If True, wrap the Zarr array in a Dask array (lazy loading). If False,
            load the entire dataset into memory as a NumPy array.

        Returns
        -------
        xr.DataArray
            The image data with proper dims, coords, and raw metadata attr.

        Notes
        -----
        * Chooses the dataset path according to `self._current_resolution_level`.
        * Attaches the original Zarr attributes under
          `constants.METADATA_UNPROCESSED`.
        """
        multiscales = self._multiscales_metadata[self._current_scene_index]
        datasets = multiscales.get("datasets", [])
        data_path = datasets[self._current_resolution_level].get("path")
        arr = self._zarr[data_path]

        if delayed:
            data = da.from_array(arr, chunks=arr.chunks)
        else:
            data = arr[:]

        coords = self._get_coords(
            list(self._get_ome_dims()),
            data.shape,
            scene=self.current_scene,
            channel_names=self.channel_names,
        )
        return xr.DataArray(
            data,
            dims=self._get_ome_dims(),
            coords=coords,
            attrs={constants.METADATA_UNPROCESSED: self.metadata},
        )

    @staticmethod
    def _get_coords(
        dims: List[str],
        shape: Tuple[int, ...],
        scene: str,
        channel_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Construct coordinate mappings for each dimension, currently only Channel.

        Parameters
        ----------
        dims : list of str
            The dimension names in order (e.g. ["T","C","Z","Y","X"]).
        shape : tuple of int
            The lengths of each dimension in `dims`.
        scene : str
            Identifier for the current scene, used in default channel IDs.
        channel_names : list of str or None
            If provided, use these names for the Channel coordinate; otherwise
            generate default OME channel IDs.

        Returns
        -------
        coords : dict
            A mapping from dimension name to coordinate values. Only includes
            entries for Channel if present in `dims`.
        """
        coords = {}
        if dimensions.DimensionNames.Channel in dims:
            # Generate channel names if no existing channel names
            if channel_names is None:
                coords[dimensions.DimensionNames.Channel] = [
                    metadata_utils.generate_ome_channel_id(scene, i)
                    for i in range(shape[dims.index(dimensions.DimensionNames.Channel)])
                ]
            else:
                coords[dimensions.DimensionNames.Channel] = channel_names
        return coords

    def _get_scale_array(self, dims: Tuple[str, ...]) -> List[float]:
        """
        Compute combined scale factors for each dimension by merging the
        overall and per-dataset coordinate transformations.

        Parameters
        ----------
        dims : tuple of str
            The dimension names in order (e.g. ("T","C","Z","Y","X")).

        Returns
        -------
        scale : list of float
            The elementwise product of the global and dataset-specific scales.
        """
        multiscales = self._multiscales_metadata[self._current_scene_index]

        if "coordinateTransformations" not in multiscales:
            log.warning(
                "No overall 'coordinateTransformations' found; "
                "defaulting overall scale to 1.0 for each dim"
            )
        overall_scale = multiscales.get(
            "coordinateTransformations", [{"scale": [1.0] * len(dims)}]
        )[0]["scale"]
        dataset_scale = multiscales["datasets"][self._current_resolution_level][
            "coordinateTransformations"
        ][0]["scale"]
        return [o * d for o, d in zip(overall_scale, dataset_scale)]

    @property
    def time_interval(self) -> Optional[types.TimeInterval]:
        """
        Returns
        -------
        sizes: Time Interval
            Using available metadata, this float represents the time interval for
            dimension T.

        """
        try:
            if dimensions.DimensionNames.Time in self._get_ome_dims():
                return self._get_scale_array(self._get_ome_dims())[
                    self._get_ome_dims().index(dimensions.DimensionNames.Time)
                ]
        except Exception as e:
            warnings.warn(f"Could not parse time interval: {e}")
        return None

    @property
    def physical_pixel_sizes(self) -> Optional[types.PhysicalPixelSizes]:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes or None
            Physical pixel sizes for Z, Y, and X if any are available;
            otherwise None. Warns and returns None on parse errors.
        """
        try:
            dims = self._get_ome_dims()
            arr = self._get_scale_array(dims)

            Z = (
                arr[dims.index(dimensions.DimensionNames.SpatialZ)]
                if dimensions.DimensionNames.SpatialZ in dims
                else None
            )
            Y = (
                arr[dims.index(dimensions.DimensionNames.SpatialY)]
                if dimensions.DimensionNames.SpatialY in dims
                else None
            )
            X = (
                arr[dims.index(dimensions.DimensionNames.SpatialX)]
                if dimensions.DimensionNames.SpatialX in dims
                else None
            )
        except Exception as e:
            warnings.warn(f"Could not parse pixel sizes: {e}")
            return None

        # If none of the spatial axes were found, return None
        if X is None and Y is None and X is None:
            return None

        return types.PhysicalPixelSizes(Z=Z, Y=Y, X=X)

    @property
    def channel_names(self) -> Optional[List[str]]:
        """
        Returns
        -------
        channel_names: List[str]
            Using available metadata, the list of strings representing channel names.
            If no channel dimension present in the data, returns None.
        """
        if self._channel_names is None:
            channels_meta = self._zarr.attrs.get("ome", {}).get("omero", {}).get(
                "channels"
            ) or self._zarr.attrs.get("omero", {}).get("channels")
            if channels_meta:
                self._channel_names = [str(ch.get("label", "")) for ch in channels_meta]
        return self._channel_names

    @property
    def metadata(self) -> GroupMetadata:
        return self._zarr.metadata

    @property
    def scale(self) -> types.Scale:
        """
        Returns
        -------
        scale: Scale
            A Scale object constructed from the Reader's time_interval and
            physical_pixel_sizes.

        Notes
        -----
        * Combines temporal and spatial scaling information into a single object.
        """
        # build a mapping from each dim → its scale value
        dims = self._get_ome_dims()
        arr = self._get_scale_array(dims)
        scale_map = dict(zip(dims, arr))

        return types.Scale(
            T=self.time_interval,
            C=scale_map.get(dimensions.DimensionNames.Channel),
            Z=scale_map.get(dimensions.DimensionNames.SpatialZ),
            Y=scale_map.get(dimensions.DimensionNames.SpatialY),
            X=scale_map.get(dimensions.DimensionNames.SpatialX),
        )

    @property
    def ome_metadata(self) -> OME:
        """
        Build multi-scene OME object with one Image per scene.
        Raises ValueError on unsupported dtype, because Pixels is required.
        """
        if hasattr(self, "_ome_metadata") and self._ome_metadata is not None:
            return self._ome_metadata

        # Dynamic PixelType lookup
        dtype_key = str(self.dtype).upper()
        try:
            pixel_type_enum = PixelType[dtype_key]
        except KeyError:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}' for Pixels.type. "
                "Cannot build OME metadata without a supported pixel type."
            )

        original_scene = self.current_scene_index
        images = []

        for idx, scene_id in enumerate(self.scenes):
            self.set_scene(idx)

            ch_meta = self._channel_metadata

            size_x = getattr(self.dims, "X", 1)
            size_y = getattr(self.dims, "Y", 1)
            size_z = getattr(self.dims, "Z", 1)
            size_c = getattr(self.dims, "C", 1)
            size_t = getattr(self.dims, "T", 1)

            channels = []
            for i in range(size_c):
                meta = ch_meta[i] if i < len(ch_meta) else {}
                contrast = []
                if not meta.get("active", True):
                    contrast.append("Off")
                if meta.get("inverted"):
                    contrast.append("inverted")
                channels.append(
                    Channel(
                        id=f"Channel:{i}",
                        name=meta.get("label", f"Channel {i}"),
                        color=meta.get("color"),
                        contrast_method=contrast or None,
                    )
                )

            pixels = Pixels(
                id=f"Pixels:{idx}",
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                size_c=size_c,
                size_t=size_t,
                type=pixel_type_enum,
                dimension_order="XYZCT",
                channels=channels,
                physical_size_x=self.scale.X,
                physical_size_y=self.scale.Y,
                physical_size_z=self.scale.Z,
                time_increment=None,
            )

            images.append(Image(id=f"Image:{idx}", name=scene_id, pixels=pixels))

        self.set_scene(original_scene)
        self._ome_metadata = OME(images=images)
        return self._ome_metadata
