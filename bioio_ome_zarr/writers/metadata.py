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
    Container for metadata inputs required to generate NGFF records.

    Parameters
    ----------
    image_name : str
        Name recorded in the multiscales metadata.
    axes : Axes
        Axes object describing names/types/units/scales/factors.
    level_shapes : Sequence[Tuple[int, ...]]
        Shapes for each pyramid level (same axis order as ``axes``).
    channels : Optional[List[Channel]]
        OMERO‑style channels. If None, channels are synthesized from shape.
    rdefs : Optional[Dict[str, Any]]
        Optional OMERO rendering defaults (e.g., {"defaultT": 0, ...}).
    creator_info : Optional[Dict[str, Any]]
        Optional creator block (only written for NGFF 0.5).
    root_transform : Optional[Dict[str, Any]]
        Optional multiscale root coordinate transformation (NGFF 0.5), e.g.:
        {"type":"scale","scale":[...]}.
    """

    image_name: str
    axes: Axes
    level_shapes: Sequence[Tuple[int, ...]]
    channels: Optional[List[Channel]] = None
    rdefs: Optional[Dict[str, Any]] = None
    creator_info: Optional[Dict[str, Any]] = None
    root_transform: Optional[Dict[str, Any]] = None


# -----------------------
# Public entry point
# -----------------------


def write_ngff_metadata(
    root: zarr.Group,
    *,
    zarr_format: int,
    params: MetadataParams,
) -> None:
    """
    Write NGFF metadata to a Zarr group.

    Parameters
    ----------
    root : zarr.Group
        Already‑opened root group to write attributes on.
    zarr_format : int
        2 → NGFF 0.4 ("multiscales" + "omero"); 3 → NGFF 0.5 ("ome").
    params : MetadataParams
        Inputs used to construct the metadata blocks.
    """
    if zarr_format == 2:
        multiscales, omero = _build_ngff_v04_from_axes_channels(params)
        root.attrs["multiscales"] = multiscales
        root.attrs["omero"] = omero
    else:
        ome_block = _build_ngff_v05_ome_block(params)
        root.attrs.update({"ome": ome_block})


# -----------------------
# Private builders
# -----------------------


def _build_ngff_v04_from_axes_channels(
    p: MetadataParams,
) -> tuple[List[dict], dict]:
    """
    Build NGFF 0.4 records for a Zarr v2 root.

    Returns
    -------
    (multiscales_list, omero_dict)
        multiscales_list : List[dict]
            Single‑entry list containing axes + per‑level dataset transforms.
        omero_dict : dict
            OMERO block with channel list and rdefs.
    """
    dims = tuple(p.axes.names)

    # 1) Axes list
    axes_list = p.axes.to_metadata()

    # 2) Per‑level datasets (scale + zero translation)
    datasets: List[Dict[str, Any]] = []
    for lvl in range(len(p.level_shapes)):
        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [
                    {"type": "scale", "scale": p.axes.scales_for_level(lvl)},
                    {"type": "translation", "translation": [0.0 for _ in dims]},
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

    # 3) Channels (use provided list or synthesize defaults by inferring C)
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

    # 4) DefaultZ for rdefs (middle plane if Z exists; else 0/1 sizing)
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


def _build_ngff_v05_ome_block(p: MetadataParams) -> dict:
    """
    Build NGFF 0.5 "ome" block for a Zarr v3 root.

    Returns
    -------
    ome : dict
        Object placed under root.attrs["ome"]. Contains a single multiscale
        entry and, optionally, an OMERO block and _creator info.
    """
    # 1) Axes
    axes_list = p.axes.to_metadata()

    # 2) Per‑level datasets (spatial scale only at dataset level)
    datasets: List[Dict[str, Any]] = []
    for lvl in range(len(p.level_shapes)):
        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [
                    {"type": "scale", "scale": p.axes.scales_for_level(lvl)}
                ],
            }
        )

    # 3) Top‑level multiscale entry (optional root transform)
    multiscale: Dict[str, Any] = {
        "name": p.image_name,
        "axes": axes_list,
        "datasets": datasets,
    }
    if p.root_transform is not None:
        multiscale["coordinateTransformations"] = [p.root_transform]

    ome: Dict[str, Any] = {"version": OME_NGFF_VERSION_V05, "multiscales": [multiscale]}

    # 4) Optional OMERO block + rdefs
    if p.channels:
        ome["omero"] = {
            "version": OME_NGFF_VERSION_V05,
            "channels": [ch.to_dict() for ch in p.channels],
        }
        if p.rdefs is not None:
            ome["omero"]["rdefs"] = p.rdefs

    # 5) Optional creator info (non‑standard convenience field)
    if p.creator_info:
        ome["_creator"] = p.creator_info

    return ome
