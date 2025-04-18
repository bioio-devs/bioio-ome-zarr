# -*- coding: utf-8 -*-

"""Top-level package for bioio_ome_zarr."""

import os
from importlib.metadata import PackageNotFoundError, version

os.environ.setdefault("ZARR_V3_EXPERIMENTAL_API", "1")

try:
    __version__ = version("bioio-ome-zarr")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "bioio-devs"
__email__ = "brian.whitney@alleninstitute.org"

from .reader import Reader
from .reader_metadata import ReaderMetadata

__all__ = ["Reader", "ReaderMetadata"]
