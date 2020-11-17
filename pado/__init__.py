__path__ = __import__("pkgutil").extend_path(__path__, __name__)
# to allow `pip install -e` to work on the `pado.ext` namespace,
# it seems we have to turn pado into a namespace too...
#
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"
