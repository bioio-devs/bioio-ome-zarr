import logging
from dataclasses import asdict
from typing import List, Optional, Tuple, Union

import dask.array as da
import numcodecs
import numpy as np
import zarr
from ngff_zarr.zarr_metadata import Axis, Dataset, Metadata, Scale, Translation
from zarr.storage import FsspecStore, LocalStore

from bioio_ome_zarr import Reader

from .utils import DimTuple, ZarrLevel, resize

log = logging.getLogger(__name__)

OME_NGFF_VERSION = "0.4"


def dim_tuple_to_dict(
    dims: Union[DimTuple, Tuple[float, float, float, float, float]]
) -> dict:
    if len(dims) != 5:
        raise ValueError("dims must be a 5-tuple in TCZYX order")
    return {k: v for k, v in zip(("t", "c", "z", "y", "x"), dims)}


def _pop_metadata_optionals(metadata_dict: dict) -> dict:
    for ax in metadata_dict.get("axes", []):
        if ax.get("unit") is None:
            ax.pop("unit", None)
    if metadata_dict.get("coordinateTransformations") is None:
        metadata_dict.pop("coordinateTransformations", None)
    return metadata_dict


def build_ome(
    size_z: int,
    image_name: str,
    channel_names: List[str],
    channel_colors: List[int],
    channel_minmax: List[Tuple[float, float]],
) -> dict:
    ch = []
    for i, name in enumerate(channel_names):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": name,
                "window": {
                    "end": float(channel_minmax[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_minmax[i][0]),
                },
            }
        )
    omero = {
        "id": 1,
        "name": image_name,
        "version": OME_NGFF_VERSION,
        "channels": ch,
        "rdefs": {"defaultT": 0, "defaultZ": size_z // 2, "model": "color"},
    }
    return omero


class OMEZarrWriter:
    """Class to write OME-Zarr files."""

    def __init__(self) -> None:
        self.output_path = ""
        self.store = None
        self.root: Optional[zarr.hierarchy.Group] = None
        self.levels: List[ZarrLevel] = []
        self._existing_level_count = 0

    def _create_level(
        self,
        idx: int,
        shape: DimTuple,
        chunk: DimTuple,
        dtype: np.dtype,
        compressor: Optional[numcodecs.abc.Codec],
    ) -> None:
        """Create one new level `idx` at `shape`/`chunk` and add to self.levels."""
        if self.root is None:
            raise RuntimeError("init_store() must be called before creating levels")
        arr = self.root.zeros(
            name=str(idx),
            shape=shape,
            chunks=chunk,
            dtype=dtype,
            compressor=compressor,
            zarr_format=2,
        )
        self.levels.append(ZarrLevel(shape, chunk, dtype, arr))

    def init_store(
        self,
        output_path: str,
        shapes: List[DimTuple],
        chunk_sizes: List[DimTuple],
        dtype: np.dtype,
        compressor: Optional[numcodecs.abc.Codec] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize the Zarr store. In overwrite mode, create all levels from scratch.
        Otherwise, append only the new levels beyond those returned by
        Reader.resolution_levels. Emits Zarr v2 metadata ('.zarray' + '.zattrs').
        """
        if len(shapes) != len(chunk_sizes) or not shapes:
            raise ValueError("shapes and chunk_sizes must align and be non-empty")

        self.output_path = output_path

        if output_path.startswith(("s3://", "gs://")):
            import fsspec

            protocol = "s3" if output_path.startswith("s3://") else "gcs"
            fs = fsspec.filesystem(protocol)
            prefix = output_path.split("://", 1)[1]
            self.store = FsspecStore(fs=fs, path=prefix)
        else:
            self.store = LocalStore(output_path)

        if overwrite:
            # OVERWRITE MODE: create all new levels
            self.root = zarr.group(
                store=self.store,
                overwrite=True,
                zarr_version=2,
            )
            self._existing_level_count = 0
            self.levels = []
            for idx, (shape, chunk) in enumerate(zip(shapes, chunk_sizes)):
                self._create_level(idx, shape, chunk, dtype, compressor)
            return
        else:
            # APPEND MODE: detect which levels already exist
            self.root = zarr.open_group(
                store=self.store,
                mode="a",
                zarr_version=2,
            )
            try:
                existing_idxs = list(Reader(self.output_path).resolution_levels)
            except Exception:
                existing_idxs = []
            self._existing_level_count = len(existing_idxs)

            # load existing levels
            self.levels = []
            for idx in existing_idxs:
                arr = zarr.open_array(store=self.store, path=str(idx), mode="a")
                shape = shapes[idx]
                chunk = chunk_sizes[idx]
                self.levels.append(ZarrLevel(shape, chunk, arr.dtype, arr))

            # create only the new levels beyond existing_idxs
            for new_idx in range(self._existing_level_count, len(shapes)):
                shape = shapes[new_idx]
                chunk = chunk_sizes[new_idx]
                self._create_level(new_idx, shape, chunk, dtype, compressor)

    def _downsample_and_write_batch_t(
        self, data_tczyx: da.Array, start_t: int, end_t: int, toffset: int = 0
    ) -> None:
        """
        Write frames [start_t, end_t) into each new level only.
        Levels with index < self._existing_level_count are skipped.
        """
        dtype = data_tczyx.dtype
        curr = data_tczyx

        for level_index, level in enumerate(self.levels):
            # downsample for levels > 0
            if level_index > 0:
                next_shape = (end_t - start_t,) + level.shape[1:]
                curr = resize(curr, next_shape, order=0).astype(dtype)

            # skip any pre-existing levels
            if level_index < self._existing_level_count:
                continue

            # write each time‐slice into the ZarrArray
            for t in range(start_t, end_t):
                slice_ = curr[[t - start_t]]
                da.to_zarr(
                    slice_,
                    level.zarray,
                    region=(slice(t + toffset, t + toffset + 1),),
                )

            log.info(f"Completed {start_t} to {end_t}")

    def write_t_batches(
        self,
        im: Reader,
        channels: List[int] = [],
        tbatch: int = 4,
        debug: bool = False,
    ) -> None:
        """
        Write the image in batches of T.

        Parameters
        ----------
        im:
            The Reader object.
        tbatch:
            The number of T to write at a time.
        """
        # loop over T in batches
        numT = im.dims.T
        if debug:
            numT = np.min([5, numT])
        log.info("Starting loop over T")
        for i in np.arange(0, numT + 1, tbatch):
            start_t = i
            end_t = min(i + tbatch, numT)
            if end_t > start_t:
                # assume start t and end t are in range (caller should guarantee this)
                ti = im.get_image_dask_data(
                    "TCZYX", T=slice(start_t, end_t), C=channels
                )
                self._downsample_and_write_batch_t(ti, start_t, end_t)
        log.info("Finished loop over T")

    def write_t_batches_image_sequence(
        self,
        paths: List[str],
        channels: List[int] = [],
        tbatch: int = 4,
        debug: bool = False,
    ) -> None:
        """
        Write the image in batches of T.

        Parameters
        ----------
        paths:
            The list of file paths, one path per T.
        tbatch:
            The number of T to write at a time.
        """
        # loop over T in batches
        numT = len(paths)
        if debug:
            numT = np.min([5, numT])
        log.info("Starting loop over T")
        for i in np.arange(0, numT + 1, tbatch):
            start_t = i
            end_t = min(i + tbatch, numT)
            if end_t > start_t:
                # read batch into dask array
                ti = []
                for j in range(start_t, end_t):
                    im = Reader(paths[j])
                    ti.append(im.get_image_dask_data("CZYX", C=channels))
                ti = da.stack(ti, axis=0)
                self._downsample_and_write_batch_t(ti, start_t, end_t)
        log.info("Finished loop over T")

    def write_t_batches_array(
        self,
        im: Union[da.Array, np.ndarray],
        channels: List[int] = [],
        tbatch: int = 4,
        toffset: int = 0,
        debug: bool = False,
    ) -> None:
        """
        Write the image in batches of T.

        Parameters
        ----------
        im:
            An ArrayLike object. Should be 5D TCZYX.
        tbatch:
            The number of T to write at a time.
        toffset:
            The offset to start writing T from. All T in the input array will be written
        """
        # if isinstance(im, (np.ndarray)):
        #     im_da = da.from_array(im)
        # else:
        #     im_da = im
        im_da = im
        # loop over T in batches
        numT = im_da.shape[0]
        if debug:
            numT = np.min([5, numT])
        log.info("Starting loop over T")
        for i in np.arange(0, numT + 1, tbatch):
            start_t = i
            end_t = min(i + tbatch, numT)
            if end_t > start_t:
                # assume start t and end t are in range (caller should guarantee this)
                ti = im_da[start_t:end_t]
                if channels:
                    for t in range(len(ti)):
                        ti[t] = [ti[t][c] for c in channels]
                self._downsample_and_write_batch_t(
                    da.asarray(ti), start_t, end_t, toffset
                )
        log.info("Finished loop over T")

    def _get_scale_ratio(self, level: int) -> Tuple[float, float, float, float, float]:
        lvl_shape = self.levels[level].shape
        lvl0_shape = self.levels[0].shape
        return (
            lvl0_shape[0] / lvl_shape[0],
            lvl0_shape[1] / lvl_shape[1],
            lvl0_shape[2] / lvl_shape[2],
            lvl0_shape[3] / lvl_shape[3],
            lvl0_shape[4] / lvl_shape[4],
        )

    def generate_metadata(
        self,
        image_name: str,
        channel_names: List[str],
        physical_dims: dict,  # {"x":0.1, "y", 0.1, "z", 0.3, "t": 5.0}
        physical_units: dict,  # {"x":"micrometer", "y":"micrometer",
        # "z":"micrometer", "t":"minute"},
        channel_colors: Union[List[str], List[int]],
    ) -> dict:
        """
        Build a metadata dict suitable for writing to ome-zarr attrs.

        Parameters
        ----------
        image_name:
            The image name.
        channel_names:
            The channel names.
        physical_dims:
            for each physical dimension, include a scale
            factor.  E.g. {"x":0.1, "y", 0.1, "z", 0.3, "t": 5.0}
        physical_units:
            For each physical dimension, include a unit
            string. E.g. {"x":"micrometer", "y":"micrometer", "z":"micrometer",
            "t":"minute"}
        """
        dims = ("t", "c", "z", "y", "x")
        axes = []
        for dim in dims:
            unit = None
            if physical_units and dim in physical_units:
                unit = physical_units[dim]
            if dim in {"x", "y", "z"}:
                axis = Axis(name=dim, type="space", unit=unit)
            elif dim == "c":
                axis = Axis(name=dim, type="channel", unit=unit)
            elif dim == "t":
                axis = Axis(name=dim, type="time", unit=unit)
            else:
                msg = f"Dimension identifier is not valid: {dim}"
                raise KeyError(msg)
            axes.append(axis)

        datasets = []
        for index, level in enumerate(self.levels):
            path = f"{index}"
            scale = []
            level_scale = self._get_scale_ratio(index)
            level_scale_dict = dim_tuple_to_dict(level_scale)
            for dim in dims:
                phys = (
                    physical_dims[dim] * level_scale_dict[dim]
                    if dim in physical_dims and dim in level_scale_dict
                    else 1.0
                )
                scale.append(phys)
            translation = []
            for dim in dims:
                # TODO handle optional translations e.g. xy stage position,
                # start time etc
                translation.append(0.0)

            coordinateTransformations = (Scale(scale), Translation(translation))
            dataset = Dataset(
                path=path, coordinateTransformations=coordinateTransformations
            )
            datasets.append(dataset)

        metadata = Metadata(
            axes=axes,
            datasets=datasets,
            name="/",
            coordinateTransformations=None,
        )
        metadata_dict = asdict(metadata)
        metadata_dict = _pop_metadata_optionals(metadata_dict)

        # get the total shape as dict:
        shapedict = dim_tuple_to_dict(self.levels[0].shape)

        # add the omero data
        ome_json = build_ome(
            shapedict["z"] if "z" in shapedict else 1,
            image_name,
            channel_names=channel_names,  # assumes we have written all channels!
            channel_colors=channel_colors,  # type: ignore
            # TODO: Rely on user to supply the per-channel min/max.
            channel_minmax=[
                (0.0, 1.0) for i in range(shapedict["c"] if "c" in shapedict else 1)
            ],
        )

        ome_zarr_metadata = {"multiscales": [metadata_dict], "omero": ome_json}
        return ome_zarr_metadata

    def write_metadata(self, metadata: dict) -> None:
        """
        Write the metadata.

        Parameters
        ----------
        metadata:
            The metadata dict. Expected to contain a multiscales
            array and omero dict
        """
        if self.root is None:
            raise RuntimeError("`init_store()` must be called before writing metadata.")
        self.root.attrs["multiscales"] = metadata["multiscales"]
        self.root.attrs["omero"] = metadata["omero"]
