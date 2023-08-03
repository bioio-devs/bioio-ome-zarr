from typing import Union


def generate_ome_image_id(image_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME image ID using a provided ID.

    Parameters
    ----------
    image_id: Union[str, int]
        A string or int representing the ID for an image.
        In the context of the usage of this function, this is usually used with the
        index of the scene / image.

    Returns
    -------
    ome_image_id: str
        The OME standard for image IDs.
    """
    return f"Image:{image_id}"


def generate_ome_channel_id(image_id: str, channel_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME channel ID using a provided ID.

    Parameters
    ----------
    image_id: str
        An image id to pull the image specific index from.
        See: `generate_ome_image_id` for more details.
    channel_id: Union[str, int]
        A string or int representing the ID for a channel.
        In the context of the usage of this function, this is usually used with the
        index of the channel.

    Returns
    -------
    ome_channel_id: str
        The OME standard for channel IDs.

    Notes
    -----
    ImageIds are usually: "Image:0", "Image:1", or "Image:N",
    ChannelIds are usually the combination of image index + channel index --
    "Channel:0:0" for the first channel of the first image for example.
    """
    # Remove the prefix 'Image:' to get just the index
    image_index = image_id.replace("Image:", "")
    return f"Channel:{image_index}:{channel_id}"


def generate_ome_instrument_id(instrument_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME instrument ID using a provided ID.

    Parameters
    ----------
    instrument_id: Union[str, int]
        A string or int representing the ID for an instrument.

    Returns
    -------
    ome_instrument_id: str
        The OME standard for instrument IDs.
    """
    return f"Instrument:{instrument_id}"


def generate_ome_detector_id(detector_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME detector ID using a provided ID.

    Parameters
    ----------
    detector_id: Union[str, int]
        A string or int representing the ID for a detector.

    Returns
    -------
    ome_detector_id: str
        The OME standard for detector IDs.
    """
    return f"Detector:{detector_id}"
