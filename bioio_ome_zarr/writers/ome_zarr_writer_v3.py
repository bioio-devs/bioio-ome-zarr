from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from .axes import Axes
from .channel import Channel
from .utils import chunk_size_from_memory_target, compute_level_shapes, resize


class OMEZarrWriterV3:
    """
    OMEZarrWriterV3 is a fully compliant OME-Zarr v0.5.0 writer built
    on Zarr v3 stores. Supports 2 ≤ N ≤ 5 dimensions (e.g. YX, ZYX,
    TYX, CZYX, or TCZYX).
    """

    def __init__(
        self,
        store: Union[str, zarr.storage.StoreLike],
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, str],
        scale_factors: Tuple[int, ...],
        axes_names: Optional[List[str]] = None,
        axes_types: Optional[List[str]] = None,
        axes_units: Optional[List[Optional[str]]] = None,
        axes_scale: Optional[List[float]] = None,
        num_levels: Optional[int] = None,
        chunk_size: Optional[Tuple[int, ...]] = None,
        shard_factor: Optional[Tuple[int, ...]] = None,
        compressor: Optional[BloscCodec] = None,
        image_name: str = "Image",
        channels: Optional[List[Channel]] = None,
        rdefs: Optional[dict] = None,
        creator_info: Optional[dict] = None,
        root_transform: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize writer and build axes + channel metadata automatically.

        Parameters
        ----------
        store : Union[str, zarr.storage.StoreLike]
            Path or Zarr store-like object for the output group.
        shape : Tuple[int, ...]
            Image shape (e.g. (2, 2), (1, 4, 3), (2, 3, 4, 5, 6)).
        dtype : Union[np.dtype, str]
            NumPy dtype of the image data (e.g. "uint8").
        scale_factors : Tuple[int, ...]
            Integer downsampling factors per axis (e.g. (1,1,2,2)).
        axes_names : Optional[List[str]]
            Names of each axis; defaults to last N of ["t","c","z","y","x"].
        axes_types : Optional[List[str]]
            Types of each axis (e.g. ["time","channel","space"]).
        axes_units : Optional[List[Optional[str]]]
            Physical units for each axis (e.g. ["ms", None, "µm"]).
        axes_scale : Optional[List[float]]
            Physical scale per axis at base resolution.
        num_levels : Optional[int]
            Number of pyramid levels to generate;
            if None, compute until no further reduction.
        chunk_size : Optional[Tuple[int,...]]
            Chunk size; None defaults to 16 mb chunk.
        shards : Optional[Tuple[int,...]]
            Shard factor; None disables sharding.
        compressor : Optional[BloscCodec]
            Zarr compressor to use (default: Blosc Zstd).
        image_name : str
            Name for the image in multiscales metadata.
        channels : Optional[List[Channel]]
            OMERO-style channel metadata objects.
        rdefs : Optional[dict]
            OMERO rendering settings (under "omero" → "rdefs").
        creator_info : Optional[dict]
            Creator metadata (e.g. {"name":"pytest","version":"0.1"}).
        root_transform : Optional[Dict[str, Any]]
            Top-level multiscale coordinate transformation
            (e.g. {"type":"scale","scale":[...]}).
        """
        # 1) Store fundamental properties
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

        # 2) Build an Axes instance (handles defaults internally)
        self.axes = Axes(
            ndim=self.ndim,
            names=axes_names,
            types=axes_types,
            units=axes_units,
            scales=axes_scale,
            factors=scale_factors,
        )

        # 3) Compute all pyramid level shapes
        self.level_shapes = compute_level_shapes(
            self.shape, self.axes.names, self.axes.factors, num_levels
        )
        # 4) Record actual number of levels
        self.num_levels = len(self.level_shapes)

        # 5) Determine uniform chunk size tuple
        if chunk_size is None:
            # Legacy auto-suggest based on base level
            self.chunk_size = chunk_size_from_memory_target(
                self.level_shapes[0], self.dtype, 16 << 20
            )
        else:
            self.chunk_size = chunk_size

        # 6) Determine uniform shard factor tuple (optional)
        self.shard_factor = shard_factor

        # 7) Initialize Zarr store and create arrays for each level
        self.root = self._init_store(store)
        self.datasets = self._create_arrays(compressor)

        # 8) Convert Channel objects to raw dicts (if provided)
        if channels is not None:
            self.channels = [ch.to_dict() for ch in channels]
        else:
            self.channels: Optional[List[Dict[str, Any]]] = None  # type: ignore

        # 9) Store OMERO render settings and top-level scale transform
        self.rdefs = rdefs
        self.root_transform = root_transform

        # 10) Write the OME-Zarr metadata block
        axes_meta = self.axes.to_metadata()
        self._write_metadata(
            name=image_name,
            axes=axes_meta,
            channels=self.channels,
            rdefs=self.rdefs,
            creator=creator_info,
            root_transform=self.root_transform,
        )

    @staticmethod
    def _init_store(store: Union[str, zarr.storage.StoreLike]) -> zarr.Group:
        """
        Create or open a Zarr group at the given store location.

        Parameters
        ----------
        store : Union[str, zarr.storage.StoreLike]
            If string containing "://", opens fsspec store in overwrite mode;
            otherwise opens a local group in overwrite mode.

        Returns
        -------
        zarr.Group
            The root group at 'store'.
        """
        if isinstance(store, str) and "://" in store:
            fs = zarr.storage.FsspecStore(store, mode="w")
            return zarr.group(store=fs, overwrite=True)
        return zarr.group(store=store, overwrite=True)

    def _create_arrays(self, resolver: Optional[BloscCodec]) -> List[zarr.Array]:
        """
        Create Zarr arrays for each multiscale level under the root group.

        Parameters
        ----------
        resolver : Optional[BloscCodec]
            If provided, uses this Blosc codec; otherwise uses default Zstd.

        Returns
        -------
        List[zarr.Array]
            One Zarr array per level, named "0", "1", etc.
        """
        comp = resolver or BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        )
        arrays: List[zarr.Array] = []
        for lvl, shape in enumerate(self.level_shapes):
            # Use uniform chunk_size and shard_factor
            chunks = self.chunk_size
            if self.shard_factor is not None:
                shards = tuple(c * self.shard_factor[i] for i, c in enumerate(chunks))
            else:
                shards = None

            arr = self.root.create_array(
                name=str(lvl),
                shape=shape,
                chunks=chunks,
                shards=shards,
                dtype=self.dtype,
                compressors=comp,
            )
            arrays.append(arr)
        return arrays

    def _write_metadata(
        self,
        name: str,
        axes: List[dict],
        channels: Optional[List[dict]],
        rdefs: Optional[dict],
        creator: Optional[dict],
        root_transform: Optional[Dict[str, Any]],
    ) -> None:
        """
        Write the 'ome' attribute with NGFF v0.5 metadata.

        Parameters
        ----------
        name : str
            Image name stored under "multiscales"[0]["name"].
        axes : List[dict]
            Axis metadata from self.axes.to_metadata().
        channels : Optional[List[dict]]
            OMERO channel metadata, under "omero" → "channels".
        rdefs : Optional[dict]
            OMERO render settings, under "omero" → "rdefs".
        creator : Optional[dict]
            Creator metadata, stored under "_creator".
        root_transform : Optional[Dict[str, Any]]
            Top-level coordinate transformation for the multiscale.

        Notes
        -----
        * Builds datasets list, each with a per-level scale transform.
        * Inserts top-level transform if provided.
        * Updates self.root.attrs['ome'].
        """
        datasets_list: List[dict] = []
        for lvl, shape in enumerate(self.level_shapes):
            scale_vals: List[float] = []
            for ax_i in range(self.ndim):
                if self.axes.types[ax_i] == "space":
                    scale_vals.append(
                        self.axes.scales[ax_i] * (self.axes.factors[ax_i] ** lvl)
                    )
                else:
                    scale_vals.append(1.0)
            datasets_list.append(
                {
                    "path": str(lvl),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale_vals}
                    ],
                }
            )

        multiscale_entry: Dict[str, Any] = {"name": name or ""}
        if root_transform is not None:
            multiscale_entry["coordinateTransformations"] = [root_transform]

        multiscale_entry["axes"] = axes
        multiscale_entry["datasets"] = datasets_list

        ome_block: Dict[str, Any] = {
            "version": "0.5",
            "multiscales": [multiscale_entry],
        }

        if channels is not None:
            omero_block: Dict[str, Any] = {"version": "0.5", "channels": channels}
            if rdefs is not None:
                omero_block["rdefs"] = rdefs
            ome_block["omero"] = omero_block

        if creator:
            ome_block["_creator"] = creator

        self.root.attrs.update({"ome": ome_block})

    def write_full_volume(self, input_data: Union[np.ndarray, da.Array]) -> None:
        """
        Write an entire image volume into the multiscale pyramid using Dask or
        NumPy input. Requires reading the full image into memory.

        Parameters
        ----------
        input_data : Union[np.ndarray, dask.array.Array]
            Full-resolution volume matching self.shape.

        Notes
        -----
        * If `input_data` is a Dask array, uses it directly; otherwise wraps the NumPy
          array with chunking at base level.
        * Uses `dask.array.store` to persist into the pre-created Zarr arrays,
          preserving chunk and shard settings.
        """
        # Wrap or reuse input as a Dask array
        if isinstance(input_data, da.Array):
            dask_array = input_data
        else:
            dask_array = da.from_array(input_data, chunks=self.chunk_size)

        # Store each pyramid level
        for lvl, shape in enumerate(self.level_shapes):
            if lvl > 0:
                dask_array = resize(dask_array, shape)
            da.store(dask_array, self.datasets[lvl], lock=True)

    def write_timepoint(
        self,
        t_index: int,
        data_t: Union[np.ndarray, da.Array],
    ) -> None:
        """
        Write a single timepoint slice into the multiscale pyramid using Dask
        or NumPy input.

        Parameters
        ----------
        t_index : int
            Index along the time axis for this slice.
        data_t : Union[np.ndarray, dask.array.Array]
            A single timepoint slice of shape self.shape[1:].

        Notes
        -----
        * Finds the "t" axis in self.axes.names.
        * Preserves uniform chunk_size and shard_factor settings on each Zarr array.
        """
        # Locate the time axis
        axis_t = next(i for i, n in enumerate(self.axes.names) if n.lower() == "t")

        # Wrap or reuse input as a Dask array with time axis
        if isinstance(data_t, da.Array):
            block = da.expand_dims(data_t, axis=axis_t)
        else:
            block = da.expand_dims(
                da.from_array(data_t, chunks=self.chunk_size[1:]),
                axis=axis_t,
            )

        # Compute and assign each pyramid level
        for lvl in range(self.num_levels):
            if lvl > 0:
                level_shape = (1,) + self.level_shapes[lvl][1:]
                # resize returns a Dask array of shape (1, ... spatial dims)
                block = resize(block, level_shape)
            # compute only this downsampled slice and drop T
            arr = block.compute()[0]

            # assign the Zarr array at the time index
            sel: List[Union[slice, int]] = [slice(None)] * self.ndim
            sel[axis_t] = t_index
            self.datasets[lvl][tuple(sel)] = arr
