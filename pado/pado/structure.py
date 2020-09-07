import enum


class PadoColumn(str, enum.Enum):
    """standardized pado columns"""

    STUDY = "study"
    ANIMAL = "animal"
    SLIDE = "slide"
    IMAGE = "image"
    ORGAN = "organ"
    FINDING = "finding"
