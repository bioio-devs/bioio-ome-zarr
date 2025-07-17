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
        shard_factor : Optional[Tuple[int,...]]
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
        self.store = store
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
        self.num_levels = len(self.level_shapes)

        # 4) Determine uniform chunk size tuple
        if chunk_size is None:
            # auto-suggest based on base level
            self.chunk_size = chunk_size_from_memory_target(
                self.level_shapes[0], self.dtype, 16 << 20
            )
        else:
            self.chunk_size = chunk_size

        # 5) Determine uniform shard factor tuple (optional)
        self.shard_factor = shard_factor
        self.compressor = compressor

        # 6) Store additional metadata fields
        self.image_name = image_name
        self.channels = [ch.to_dict() for ch in channels] if channels else None
        self.rdefs = rdefs
        self.creator_info = creator_info
        self.root_transform = root_transform

        # 7) Placeholder for store and dataset references
        self.root: Optional[zarr.Group] = None
        self.datasets: List[zarr.Array] = []
        self._initialized: bool = False

    def _initialize(self) -> None:
        """
        Create the Zarr store and arrays, and write multiscale metadata.
        Should only be called once, lazily.
        """
        if isinstance(self.store, str) and "://" in self.store:
            fs = zarr.storage.FsspecStore(self.store, mode="w")
            self.root = zarr.group(store=fs, overwrite=True)
        else:
            self.root = zarr.group(store=self.store, overwrite=True)

        comp = self.compressor or BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        )
        self.datasets = []
        for lvl, shape in enumerate(self.level_shapes):
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
            self.datasets.append(arr)

        self._write_metadata()
        self._initialized = True

    def _write_metadata(self) -> None:
        """
        Write NGFF v0.5 multiscale and OMERO metadata to root.attrs["ome"].
        """
        if self.root is None:
            raise RuntimeError("Store must be initialized before writing metadata.")

        axes_meta = self.axes.to_metadata()
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

        multiscale_entry: Dict[str, Any] = {"name": self.image_name or ""}
        if self.root_transform is not None:
            multiscale_entry["coordinateTransformations"] = [self.root_transform]

        multiscale_entry["axes"] = axes_meta
        multiscale_entry["datasets"] = datasets_list

        ome_block: Dict[str, Any] = {
            "version": "0.5",
            "multiscales": [multiscale_entry],
        }

        if self.channels is not None:
            omero_block: Dict[str, Any] = {"version": "0.5", "channels": self.channels}
            if self.rdefs is not None:
                omero_block["rdefs"] = self.rdefs
            ome_block["omero"] = omero_block

        if self.creator_info:
            ome_block["_creator"] = self.creator_info

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
        if not self._initialized:
            self._initialize()

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
        if not self._initialized:
            self._initialize()

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
