import enum


class PadoColumn(str, enum.Enum):
    """standardized pado columns"""

    STUDY = "study"
    COMPOUND = "compound"
    ANIMAL = "animal"
    SLIDE = "slide"
    IMAGE = "image"
    ORGAN = "organ"
    FINDING = "finding"

    def __str__(self):
        return str.__str__(self)
