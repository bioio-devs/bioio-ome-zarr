from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import zarr

from .axes import Axes
from .channel import Channel

OME_NGFF_VERSION_V04 = "0.4"
OME_NGFF_VERSION_V05 = "0.5"


@dataclass
class MetadataParams:
    """
    Parameters required to render NGFF metadata.

    Fields
    ------
    image_name : str
        Display name of the image.
    axes : Axes
        Axes descriptor (names/types/units/scales).
    level_shapes : Sequence[Tuple[int, ...]]
        Per-level array shapes, including level 0.
    channels : Optional[List[Channel]]
        OMERO-style channel metadata; if None, defaults are inferred.
    rdefs : Optional[Dict[str, Any]]
        OMERO rendering defaults (placed under ome->omero->rdefs for 0.5).
    creator_info : Optional[Dict[str, Any]]
        Optional creator block stored under ome->_creator (0.5).
    root_transform : Optional[Dict[str, Any]]
        Optional transform at the multiscale root.
    dataset_scales : Optional[List[List[float]]]
        For levels > 0, per-axis *relative size vs. level 0*.
        Example: for level 1 where spatial dims halve, use [1,1,1,0.5,0.5].
        If None, only level 0 is expected.
    """

    image_name: str
    axes: Axes
    level_shapes: Sequence[Tuple[int, ...]]
    channels: Optional[List[Channel]] = None
    rdefs: Optional[Dict[str, Any]] = None
    creator_info: Optional[Dict[str, Any]] = None
    root_transform: Optional[Dict[str, Any]] = None
    dataset_scales: Optional[List[List[float]]] = None


def write_ngff_metadata(
    root: zarr.Group,
    *,
    zarr_format: int,
    params: MetadataParams,
) -> None:
    """
    Write NGFF metadata to the Zarr group attributes.

    For Zarr v2: writes NGFF 0.4 `multiscales` and `omero`.
    For Zarr v3: writes NGFF 0.5 `ome` with `multiscales` (and optional `omero`).

    Parameters
    ----------
    root : zarr.Group
        Opened Zarr group to receive attributes.
    zarr_format : int
        2 for NGFF 0.4 layout; 3 for NGFF 0.5.
    params : MetadataParams
        Metadata inputs (axes, level shapes, channels, etc.).
    """
    if zarr_format == 2:
        multiscales, omero = _build_ngff_v04(params)
        root.attrs["multiscales"] = multiscales
        root.attrs["omero"] = omero
    else:
        ome_block = _build_ngff_v05(params)
        root.attrs.update({"ome": ome_block})


def _level_scale_from_dataset_scales(
    axes: Axes,
    lvl: int,
    dataset_scales: Optional[List[List[float]]],
) -> List[float]:
    """
    Compute per-axis scale transform for a given level.
    If `dataset_scales` is None, only level 0 is expected; spatial axes use
    `axes.scales[i]` (default 1.0), and non-spatial axes use 1.0.
    """
    out: List[float] = []
    for i, ax_type in enumerate(axes.types):
        base = float(axes.scales[i] if i < len(axes.scales) else 1.0)
        if ax_type == "space":
            if lvl == 0 or dataset_scales is None:
                out.append(base)
            else:
                rel_size = float(dataset_scales[lvl - 1][i])
                rel_size = rel_size if rel_size != 0.0 else 1.0
                out.append(base * (1.0 / rel_size))
        else:
            out.append(1.0)
    return out


def _build_ngff_v04(p: MetadataParams) -> tuple[List[dict], dict]:
    """
    Build NGFF 0.4 `multiscales` and `omero` dicts.
    """
    # Axes list (name/type and optional unit)
    axes_list = p.axes.to_metadata()

    # Per-level datasets with scale + zero translation
    datasets = []
    for lvl in range(len(p.level_shapes)):
        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": _level_scale_from_dataset_scales(
                            p.axes, lvl, p.dataset_scales
                        ),
                    },
                    {"type": "translation", "translation": [0.0 for _ in p.axes.names]},
                ],
            }
        )

    multiscales = [
        {
            "axes": axes_list,
            "datasets": datasets,
            "name": "/",
            "version": OME_NGFF_VERSION_V04,
        }
    ]

    # OMERO
    if p.channels:
        channel_list = [ch.to_dict() for ch in p.channels]
    else:
        try:
            c_axis = p.axes.index_of("c")
            C = int(p.level_shapes[0][c_axis])
        except ValueError:
            C = 1
        channel_list = [
            Channel(label=f"C:{i}", color="ffffff").to_dict() for i in range(C)
        ]

    # defaultZ for rdefs
    try:
        z_axis = p.axes.index_of("z")
        size_z = int(p.level_shapes[0][z_axis])
    except ValueError:
        size_z = 1

    omero = {
        "id": 1,
        "name": p.image_name,
        "version": OME_NGFF_VERSION_V04,
        "channels": channel_list,
        "rdefs": {"defaultT": 0, "defaultZ": size_z // 2, "model": "color"},
    }
    return multiscales, omero


def _build_ngff_v05(p: MetadataParams) -> dict:
    """
    Build NGFF 0.5 `ome` block with multiscales and optional OMERO.
    """
    # axes
    axes_list = p.axes.to_metadata()

    # datasets: per-level spatial scale only
    datasets = []
    for lvl in range(len(p.level_shapes)):
        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": _level_scale_from_dataset_scales(
                            p.axes, lvl, p.dataset_scales
                        ),
                    }
                ],
            }
        )

    multiscale = {"name": p.image_name, "axes": axes_list, "datasets": datasets}
    if p.root_transform is not None:
        multiscale["coordinateTransformations"] = [p.root_transform]

    ome: Dict[str, Any] = {"version": OME_NGFF_VERSION_V05, "multiscales": [multiscale]}

    # Omero
    if p.channels:
        ome["omero"] = {
            "version": OME_NGFF_VERSION_V05,
            "channels": [ch.to_dict() for ch in p.channels],
        }
        if p.rdefs is not None:
            ome["omero"]["rdefs"] = p.rdefs

    if p.creator_info:
        ome["_creator"] = p.creator_info

    return ome
