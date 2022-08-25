from __future__ import annotations

import importlib
import warnings

_COMPAT = {
    "ImageId": "pado.images.ids",
    "Image": "pado.images.image",
    "ImageProvider": "pado.images.providers",
    "FilteredImageProvider": "pado.images.providers",
    "GroupedImageProvider": "pado.images.providers",
    "Tile": "pado.images.tiles",
    "TileIterator": "pado.images.tiles",
}


def __getattr__(name):
    try:
        module = _COMPAT[name]
    except KeyError:
        raise AttributeError(name)
    else:
        warnings.warn(
            f"{__name__}.{name} moved to {module}.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module), name)
