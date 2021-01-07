__path__ = __import__("pkgutil").extend_path(__path__, __name__)
# to allow `pip install -e` to work on the `pado.ext` namespace,
# it seems we have to turn pado into a namespace too...
#
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

# noinspection PyUnresolvedReferences
__all__ = [
    "PadoDataset",
    "PadoDatasetChain",
]


# allow importing items in __all__
def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return getattr(import_module("pado.dataset"), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
