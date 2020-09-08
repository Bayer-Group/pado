import enum
import string
from typing import Iterable

__all__ = ["PadoColumn", "PadoReserved"]

ALLOWED_CHARACTERS = set(string.ascii_letters + string.digits + "_")
SEPARATOR = "__"


class PadoReserved(str, enum.Enum):
    """reserved pado columns"""

    __version__ = 1
    __str__ = str.__str__

    DATA_SOURCE_ID = "_pado_data_source_id"
    IMAGE_REL_PATH = "_pado_image_rel_path"


class PadoInvalid(str, enum.Enum):
    """invalid pado columns"""

    __version__ = 1
    __str__ = str.__str__

    RESERVED_COL_INDEX = "level_0"


class _SubColumnMixin:
    """mixin allowing to add sub columns to standardized columns"""

    def subcolumn(self, subcolumn: str):
        if "__" in subcolumn:
            raise ValueError("cannot contain double underscore")
        if not ALLOWED_CHARACTERS.issuperset(subcolumn):
            raise ValueError("can only contain numbers letters and underscore")
        if subcolumn.startswith("_") or subcolumn.endswith("_"):
            raise ValueError("may not start or end with an underscore")
        if subcolumn in set(PadoReserved):
            raise ValueError("cannot use reserved string")
        if subcolumn in set(PadoInvalid):
            raise ValueError("cannot use invalid string")
        return SEPARATOR.join([self, subcolumn.upper()])


class PadoColumn(_SubColumnMixin, str, enum.Enum):
    """standardized pado columns"""

    __version__ = 1
    __str__ = str.__str__

    STUDY = "STUDY"
    EXPERIMENT = "EXPERIMENT"
    ANIMAL_GROUP = "GROUP"
    ANIMAL = "ANIMAL"
    COMPOUND = "COMPOUND"
    ORGAN = "ORGAN"
    SLIDE = "SLIDE"
    IMAGE = "IMAGE"
    FINDING = "FINDING"


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
