"""pado: the pragmatic pathology dataset library"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

if TYPE_CHECKING:
    from pado.dataset import PadoDataset

__all__ = [
    "PadoDataset",
]


# allow importing items in __all__
def __getattr__(name):
    from importlib import import_module

    if name in __all__:
        return getattr(import_module("pado.dataset"), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
