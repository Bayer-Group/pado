from abc import ABC

# TODO:
#   this needs more thought...
#   my first test implementation didn't feel correct...
#   - add class registry in pado?
#
class DataSource(ABC):
    """DataSource base class

    All data sources should go through this abstraction to
    allow channelling them into the same output format.

    """
    pass
