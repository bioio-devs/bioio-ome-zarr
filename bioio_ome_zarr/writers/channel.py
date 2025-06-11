from typing import Any, Dict, Optional


class Channel:
    """
    Helper to construct an OMERO-style channel metadata dict, compliant with
    the NGFF OME-Zarr 0.5.0 OMERO block specification
    (see https://ngff.openmicroscopy.org/0.5/#omero-md).

    Only `label` and `color` are required; all other parameters have sensible defaults.
    Window defaults to 0–255.
    """

    def __init__(
        self,
        *,
        label: str,
        color: str,
        active: bool = True,
        coefficient: float = 1.0,
        family: str = "linear",
        inverted: bool = False,
        window: Optional[Dict[str, int]] = None,
    ):
        """
        Parameters
        ----------
        label : str
            Channel label (e.g. "AF488-T2").
        color : str
            Hex color code (e.g. "00FF00").
        active : bool
            Whether the channel is active (default True).
        coefficient : float
            Color coefficient (default 1.0).
        family : str
            Interpolation family (default "linear").
        inverted : bool
            Whether to invert the channel (default False).
        window : Optional[Dict[str,int]]
            If provided, must contain keys "min","max","start","end".
            Otherwise defaults to {"min":0, "max":255, "start":0, "end":255}.
        """
        self.label = label
        self.color = color
        self.active = active
        self.coefficient = coefficient
        self.family = family
        self.inverted = inverted

        if window is not None:
            self.window = window
        else:
            # Default: 0–255
            self.window = {"min": 0, "max": 255, "start": 0, "end": 255}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to the raw dict form expected by OME-Zarr’s OMERO block.
        """
        return {
            "color": self.color,
            "coefficient": self.coefficient,
            "active": self.active,
            "label": self.label,
            "window": {
                "min": self.window["min"],
                "max": self.window["max"],
                "start": self.window["start"],
                "end": self.window["end"],
            },
            "family": self.family,
            "inverted": self.inverted,
        }
