# bioio-ome-zarr

[![Build Status](https://github.com/bioio-devs/bioio-ome-zarr/workflows/CI/badge.svg)](https://github.com/bioio-devs/bioio-ome-zarr/actions)
[![Documentation](https://github.com/bioio-devs/bioio-ome-zarr/workflows/Documentation/badge.svg)](https://bioio-devs.github.io/bioio-ome-zarr)

A BioIO reader plugin for reading Zarr files in the OME format.

This plugin is intended to be used in conjunction with [bioio](https://github.com/bioio-devs/bioio)
---

## Installation

**Stable Release:** `pip install bioio-ome-zarr`<br>
**Development Head:** `pip install git+https://github.com/BrianWhitneyAI/bioio-ome-zarr.git`

## Quickstart

```python
from bioio_ome_zarr import Reader 

r = Reader("my-image.zarr")
r.dims
```

## Documentation

For full package documentation please visit [BrianWhitneyAI.github.io/bioio-ome-zarr](https://BrianWhitneyAI.github.io/bioio-ome-zarr).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
