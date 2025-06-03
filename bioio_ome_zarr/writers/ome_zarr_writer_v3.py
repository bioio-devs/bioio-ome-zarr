import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from .axes import Axes
from .channel import Channel


def spatial_downsample(
    data: np.ndarray,
    axes_names: List[str],
    scale_factors: Tuple[int, ...],
) -> np.ndarray:
    """
    Downsample `data` only over X and Y axes (axes named "x" or "y"),
    by their corresponding factor in `scale_factors`. All other axes unchanged.
    """
    names_lower = [n.lower() for n in axes_names]
    shape = data.shape

    crop_slices: List[Union[slice, int]] = []
    reshape_dims: List[int] = []
    factor_axes: List[int] = []
    new_dim_idx = 0

    for i, dim_size in enumerate(shape):
        ax_name = names_lower[i]
        f = scale_factors[i]

        if ax_name in ("x", "y") and f > 1 and dim_size >= f:
            full = (dim_size // f) * f
            crop_slices.append(slice(0, full))
            reshape_dims.extend([full // f, f])
            factor_axes.append(new_dim_idx + 1)
            new_dim_idx += 2
        else:
            crop_slices.append(slice(None))
            reshape_dims.append(dim_size)
            new_dim_idx += 1

    data_cropped = data[tuple(crop_slices)]
    data_reshaped = data_cropped.reshape(tuple(reshape_dims))
    downsampled = data_reshaped.mean(axis=tuple(factor_axes))

    return downsampled


class OMEZarrWriterV3:
    """
    OMEZarrWriterV3 is a fully compliant OME-Zarr v0.5.0 writer built
    on Zarr v3 stores. It supports writing inputs of dimensionality
    2 ≤ N ≤ 5 (e.g., YX, ZYX, TYX, CZYX, or TCZYX).
    """

    def __init__(
        self,
        store: Union[str, zarr.storage.StoreLike],
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, str],
        axes_names: Optional[List[str]] = None,
        axes_types: Optional[List[str]] = None,
        axes_units: Optional[List[Optional[str]]] = None,
        axes_scale: Optional[List[float]] = None,
        scale_factors: Optional[Tuple[int, ...]] = None,
        num_levels: Optional[int] = None,
        chunks: Union[str, Tuple[int, ...], List[Tuple[int, ...]]] = "auto",
        shards: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        compressor: Optional[BloscCodec] = None,
        image_name: str = "Image",
        channels: Optional[List[Channel]] = None,
        rdefs: Optional[dict] = None,
        creator_info: Optional[dict] = None,
        multiscale_scale: Optional[List[float]] = None,
    ):
        """
        Initialize writer and build axes + channel metadata automatically.

        Added parameter:
          multiscale_scale: Optional[List[float]]
            If provided, will be inserted as a top-level
            scale transform under each multiscale entry.
        """
        # 1) Store fundamental properties
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

        # 2) Build an Axes instance (handles default slicing internally)
        self.axes = Axes(
            ndim=self.ndim,
            names=axes_names,
            types=axes_types,
            units=axes_units,
            scales=axes_scale,
            factors=scale_factors,
        )

        # 3) Compute all possible pyramid level shapes
        self.level_shapes = self._compute_levels(num_levels)

        # 4) Record actual number of levels
        self.num_levels = len(self.level_shapes)

        # 5) Prepare chunk & shard parameters per level
        self.chunks = self._prepare_parameter(chunks, self._suggest_chunks, "chunks")
        self.shards = self._prepare_parameter(
            shards, lambda s: s, "shards", required=False
        )

        # 6) Initialize Zarr store and create arrays for each level
        self.root = self._init_store(store)
        self.datasets = self._create_arrays(compressor)

        # 7) Convert Channel objects to raw dicts (if provided)
        if channels is not None:
            self.channels = [ch.to_dict() for ch in channels]
        else:
            self.channels: Optional[List[Dict[str, Any]]] = None  # type: ignore

        # 8) Store OMERO render settings verbatim
        self.rdefs = rdefs

        # 9) Capture optional top‐level scale transform
        self.multiscale_scale = multiscale_scale

        # 10) Build axes metadata list from the Axes helper
        axes_meta = self.axes.to_metadata()

        # 11) Write the OME-Zarr metadata (multiscales + OMERO + creator)
        self._write_metadata(
            name=image_name,
            axes=axes_meta,
            channels=self.channels,
            rdefs=self.rdefs,
            creator=creator_info,
        )

    def _compute_levels(self, max_levels: Optional[int]) -> List[Tuple[int, ...]]:
        """
        Calculate multiresolution level shapes by repeatedly scaling X and Y axes.
        Only X and Y axes get downsampled; all other axes remain fixed.
        """
        shapes: List[Tuple[int, ...]] = [self.shape]
        lvl = 1

        while max_levels is None or lvl < max_levels:
            prev = shapes[-1]
            nxt: List[int] = []
            for i, size in enumerate(prev):
                ax_name = self.axes.names[i].lower()
                factor = self.axes.factors[i]
                if ax_name in ("x", "y") and factor > 1:
                    nxt.append(max(1, size // factor))
                else:
                    nxt.append(size)
            nxt_tuple = tuple(nxt)
            if nxt_tuple == prev:
                break
            shapes.append(nxt_tuple)
            lvl += 1

        return shapes

    def _prepare_parameter(
        self,
        param: Any,
        default_fn: Callable[[Tuple[int, ...]], Tuple[int, ...]],
        name: str,
        required: bool = True,
    ) -> List[Optional[Tuple[int, ...]]]:
        """
        Standardize chunk or shard specification across levels.
        """
        levels = len(self.level_shapes)

        if param == "auto" and name == "chunks":
            return [default_fn(s) for s in self.level_shapes]

        if param is None:
            if not required:
                return [None] * levels
            default = default_fn(self.level_shapes[0])
            return [default] * levels

        if isinstance(param, list):
            items = param
        else:
            items = [param] * levels

        return [
            tuple(min(items[i][d], self.level_shapes[i][d]) for d in range(self.ndim))
            for i in range(levels)
        ]

    @staticmethod
    def _init_store(store: Union[str, zarr.storage.StoreLike]) -> zarr.Group:
        """
        Create or open a Zarr group at the given store location.
        """
        if isinstance(store, str) and "://" in store:
            fs = zarr.storage.FsspecStore(store, mode="w")
            return zarr.group(store=fs, overwrite=True)
        return zarr.group(store=store, overwrite=True)

    def _create_arrays(self, resolver: Optional[BloscCodec]) -> List[zarr.Array]:
        """
        Create Zarr arrays for each multiscale level in the root group.
        """
        comp = resolver or BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        )
        arrays: List[zarr.Array] = []
        for lvl, shape in enumerate(self.level_shapes):
            chunks_lvl = self.chunks[lvl]
            shards_lvl = self.shards[lvl]
            if chunks_lvl is not None and shards_lvl is not None:
                shards_param = tuple(c * s for c, s in zip(chunks_lvl, shards_lvl))
            else:
                shards_param = None

            arr = self.root.create_array(
                name=str(lvl),
                shape=shape,
                chunks=chunks_lvl,
                shards=shards_param,
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
    ) -> None:
        """
        Write the 'ome' attribute on the root group with NGFF v0.5 metadata.
        """
        # Build per‐level datasets list
        datasets_list: List[dict] = []
        for lvl in range(self.num_levels):
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

        # Build the multiscale entry
        multiscale_entry: Dict[str, Any] = {"name": name or ""}

        # If a top-level scale is provided, insert it here:
        if self.multiscale_scale is not None:
            multiscale_entry["coordinateTransformations"] = [
                {"type": "scale", "scale": self.multiscale_scale}
            ]

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

    def _suggest_chunks(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Suggest chunk shapes aiming for ~64MB per chunk, based on dtype and shape.
        Only "space" axes get larger chunks; non-"space" axes → 1.
        """
        bpe = self.dtype.itemsize
        maxe = (64 << 20) // bpe  # max elements ≈ 64MB

        spatial_idxs = [i for i, t in enumerate(self.axes.types) if t == "space"]
        chunk = [1] * self.ndim

        if len(spatial_idxs) == 1:
            i0 = spatial_idxs[0]
            chunk[i0] = min(shape[i0], maxe)

        elif len(spatial_idxs) == 2:
            iY, iX = spatial_idxs[-2], spatial_idxs[-1]
            y_size, x_size = shape[iY], shape[iX]
            base = int(math.sqrt(maxe))
            chunk_x = min(x_size, base)
            chunk_y = min(y_size, max(1, maxe // chunk_x))
            chunk[iX] = chunk_x
            chunk[iY] = chunk_y

        elif len(spatial_idxs) == 3:
            iZ, iY, iX = spatial_idxs[-3], spatial_idxs[-2], spatial_idxs[-1]
            z_size, y_size, x_size = shape[iZ], shape[iY], shape[iX]
            base = int(math.sqrt(maxe))
            chunk_x = min(x_size, base)
            chunk_y = min(y_size, max(1, maxe // chunk_x))
            chunk_z = min(z_size, max(1, maxe // (chunk_x * chunk_y)))
            chunk[iX] = chunk_x
            chunk[iY] = chunk_y
            chunk[iZ] = chunk_z

        return tuple(chunk)

    def write_full_volume(self, data: np.ndarray) -> None:
        """
        Write an entire image volume into the multiscale pyramid.
        """
        self.datasets[0][...] = data
        cur = data
        for lvl in range(1, self.num_levels):
            cur = spatial_downsample(cur, self.axes.names, self.axes.factors)
            self.datasets[lvl][...] = cur

    def write_timepoint(self, t_index: int, data_t: np.ndarray) -> None:
        """
        Write a single timepoint slice into the multiscale pyramid.
        """
        # Find the index of the "t" axis
        axis_t = [i for i, n in enumerate(self.axes.names) if n.lower() == "t"][0]

        # Build a selector for level 0: List[Union[slice, int]]
        sel0: List[Union[slice, int]] = [slice(None)] * self.ndim
        sel0[axis_t] = t_index  # type: ignore[index]  # int into Union[slice,int]
        self.datasets[0][tuple(sel0)] = data_t

        # Expand dims at the time axis for downsampling
        block = np.expand_dims(data_t, axis=axis_t)

        for lvl in range(1, self.num_levels):
            block = spatial_downsample(block, self.axes.names, self.axes.factors)

            # Build a selector to extract the single frame from the expanded block
            sl_sngl: List[Union[slice, int]] = [slice(None)] * self.ndim
            sl_sngl[axis_t] = 0  # type: ignore[index]
            slice_data = block[tuple(sl_sngl)]

            # Now build a selector to place that slice into level 'lvl'
            sel_lvl: List[Union[slice, int]] = [slice(None)] * self.ndim
            sel_lvl[axis_t] = t_index  # type: ignore[index]
            self.datasets[lvl][tuple(sel_lvl)] = slice_data
