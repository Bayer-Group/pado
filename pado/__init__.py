__path__ = __import__("pkgutil").extend_path(__path__, __name__)
# to allow `pip install -e` to work on the `pado.ext` namespace,
# it seems we have to turn pado into a namespace too...
# datasource modules and cannot be modified.
#
# What we actually want is have the below code in pado/__init__.py
# so that plugin modules can't break the `pado.` module.
# But PyCharm doesn't play nice then. So for the sake of smooth
# development we give up on pado.__version__ for now.
#
# try:
#     from ._version import version as __version__
# except ImportError:
#     __version__ = "not-installed"
