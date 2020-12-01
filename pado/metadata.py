import enum
import string
from typing import Iterable

import pandas as pd

__all__ = [
    "PadoColumn",
    "PadoInvalid",
    "PadoReserved",
    "verify_columns",
]

ALLOWED_CHARACTERS = set(string.ascii_letters + string.digits + "_")
SEPARATOR = "__"


class PadoReserved(str, enum.Enum):
    """reserved pado columns"""

    __version__ = 1

    DATA_SOURCE_ID = "_pado_data_source_id"
    IMAGE_REL_PATH = "_pado_image_rel_path"

    def __str__(self):
        return str(self.value)


class PadoInvalid(str, enum.Enum):
    """invalid pado columns"""

    __version__ = 1

    RESERVED_COL_INDEX = "level_0"

    def __str__(self):
        return str(self.value)


class PadoColumn(str, enum.Enum):
    """standardized pado columns"""

    __version__ = 2

    SOURCE = "SOURCE"
    STUDY = "STUDY"
    EXPERIMENT = "EXPERIMENT"
    GROUP = "GROUP"
    ANIMAL = "ANIMAL"
    COMPOUND = "COMPOUND"
    ORGAN = "ORGAN"
    SLIDE = "SLIDE"
    IMAGE = "IMAGE"
    FINDING = "FINDING"

    def __str__(self):
        return str(self.value)

    def subcolumn(self, subcolumn: str):
        if "__" in subcolumn:
            raise ValueError("cannot contain double underscore")
        if not ALLOWED_CHARACTERS.issuperset(subcolumn):
            raise ValueError("can only contain numbers letters and underscore")
        if subcolumn in set(PadoReserved):
            raise ValueError("cannot use reserved string")
        if subcolumn in set(PadoInvalid):
            raise ValueError("cannot use invalid string")
        if subcolumn.startswith("_") or subcolumn.endswith("_"):
            raise ValueError("may not start or end with an underscore")
        return SEPARATOR.join([self, subcolumn.upper()])


def verify_columns(columns: Iterable[str], raise_if_invalid: bool = True) -> bool:
    """verify that columns are pado compatible"""
    invalid = []
    for column in columns:
        if column in set(PadoReserved):
            continue
        elif column in set(PadoInvalid):
            invalid.append(column)
        elif not ALLOWED_CHARACTERS.issuperset(column):
            invalid.append(column)
        elif not column.startswith(tuple(PadoColumn)):
            invalid.append(column)
        elif "___" in column:  # TODO: use SEPARATOR to do overlap matching?
            invalid.append(column)
    if invalid:
        if raise_if_invalid:
            raise ValueError(f"found invalid columns: {invalid}")
        return False
    return True


@pd.api.extensions.register_dataframe_accessor("pado")
class PadoAccessor:
    """provide pado specific operations on the dataframe"""

    c = PadoColumn
    """provide shorthand for standardized columns"""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._df = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame):
        """validate the provided dataframe"""
        # check required columns
        req = set(PadoColumn)
        # so not require SOURCE
        req.remove("SOURCE")  # FIXME: need to revisit when columns get revisited
        if not req.issubset(obj.columns):
            missing = set(PadoColumn) - set(obj.columns)
            mc = ", ".join(map(str.__repr__, sorted(missing)))
            raise AttributeError(f"missing columns: {mc}")
        # check if columns are compliant
        try:
            verify_columns(columns=obj.columns, raise_if_invalid=True)
        except ValueError as err:
            raise AttributeError(str(err))

    def _subset(self, column: PadoColumn) -> pd.DataFrame:
        """return the dataframe subset belonging to a PadoColumn"""
        return self._df.loc[:, self._cm[column]].drop_duplicates()

    class _SubsetDescriptor:
        """descriptor for accessing the dataframe subsets"""

        def __init__(self, pado_column: PadoColumn):
            self._col = pado_column

        def __get__(self, instance, owner):
            if instance is None:
                return self  # pragma: no cover
            # noinspection PyProtectedMember
            return instance._subset(self._col)

    # the dataframe accessors
    studies = _SubsetDescriptor(c.STUDY)
    experiments = _SubsetDescriptor(c.EXPERIMENT)
    groups = _SubsetDescriptor(c.GROUP)
    animals = _SubsetDescriptor(c.ANIMAL)
    compounds = _SubsetDescriptor(c.COMPOUND)
    organs = _SubsetDescriptor(c.ORGAN)
    slides = _SubsetDescriptor(c.SLIDE)
    images = _SubsetDescriptor(c.IMAGE)
    findings = _SubsetDescriptor(c.FINDING)
