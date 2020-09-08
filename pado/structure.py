import enum


class PadoReserved(str, enum.Enum):
    """reserved pado columns"""

    __str__ = str.__str__

    SRC_ID = "_pado_source_id"
    REL_PATH = "_pado_rel_path"
    _RESERVED_COL_INDEX = "level_0"


class PadoColumn(str, enum.Enum):
    """standardized pado columns"""

    __str__ = str.__str__

    STUDY = "study"
    COMPOUND = "compound"
    ANIMAL = "animal"
    SLIDE = "slide"
    IMAGE = "image"
    ORGAN = "organ"
    FINDING = "finding"
