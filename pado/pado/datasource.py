from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

# fixme: this file is junk

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

    @abstractmethod
    def retrieve_raw(self) -> Any:
        """retrieve the data source in raw form"""
        return None

    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        """return the data as a dataframe"""
        raise NotImplementedError("implement in subclass")
