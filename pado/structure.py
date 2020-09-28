import enum
import string
from collections import defaultdict
from typing import Dict, Iterable, List

import pandas as pd

__all__ = [
    "PadoColumn",
    "PadoInvalid",
    "PadoReserved",
    "build_column_map",
    "structurize_metadata",
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


def build_column_map(columns: Iterable[str]) -> Dict[str, List[str]]:
    """build a map of subcolumns"""
    output = defaultdict(list)
    ignore = set(PadoReserved)
    valid = set(PadoColumn)
    for col in columns:
        key, _, _ = col.partition(SEPARATOR)
        if key not in ignore:
            output[key].append(str(col))
    if not (output.keys() <= valid):  # Set comparison
        raise ValueError(f"unsupported keys {set(output) - set(PadoColumn)}")
    output.default_factory = None
    return output


def _collapse_dataframe_to_dict(df: pd.DataFrame, structure: Dict, col_map: Dict):
    """helper function to collapse the dataframe into a dictionary"""
    # flatten
    columns = []
    for toplevel in structure.values():
        columns.extend(col_map[toplevel])

    sdf = df[columns].drop_duplicates()
    if len(sdf) > 1:
        raise ValueError("can't collapse...")

    md = {}
    for md_key_tuple, toplevel in structure.items():
        d = md
        for md_key in md_key_tuple:
            d = md.setdefault(md_key, {})

        for col in col_map[toplevel]:
            key, _, subkey = col.partition(SEPARATOR)
            if subkey == "":
                subkey = col
            d[subkey] = sdf[col].item()

    return md


def structurize_metadata(
    df: pd.DataFrame, root_column: PadoColumn, col_map: Dict
) -> Dict:
    """try to enforce sensible relationship structure

    FIXME:
      The relationship between the toplevel "physical" objects in the pathology
      data should be defined somewhere. For the sake of moving fast we implement
      it one by one for what we need at the time. This should be replaced...

    """
    if root_column != PadoColumn.IMAGE:
        raise NotImplementedError("relationship structure not implemented yet")

    # Enforce the structure for the metadata:
    # TODO: verify if this is sensible
    # IMAGE M->1 SLIDE
    #   SLIDE M->N ORGAN
    #     ORGAN 1->1 ANIMAL
    #       ANIMAL M->1 GROUP
    #         GROUP M->1 EXPERIMENT
    #           EXPERIMENT M->1 COMPOUND
    #           EXPERIMENT M->1 STUDY
    #     ORGAN 1->M FINDING
    #
    # This means, that using IMAGE as root we should be able to do:
    # image_md = {
    #   "slide": {
    #     "organ": [{
    #       "animal": {
    #         "group": {
    #           "experiment": {
    #             "compound": {},
    #             "study": {},
    #           },
    #         },
    #       },
    #       "finding": [
    #         ...
    #       ]
    #     }, ...
    #     ]
    #   }

    structure = {
        (): PadoColumn.IMAGE,
        ("slide",): PadoColumn.SLIDE,
    }
    md = _collapse_dataframe_to_dict(df, structure, col_map)
    md["slide"]["organ"] = []

    for _, odf in df.groupby([PadoColumn.ORGAN, PadoColumn.ANIMAL]):
        organ_structure = {
            (): PadoColumn.ORGAN,
            ("animal",): PadoColumn.ANIMAL,
            ("animal", "group"): PadoColumn.GROUP,
            ("animal", "group", "experiment"): PadoColumn.EXPERIMENT,
            ("animal", "group", "experiment", "compound"): PadoColumn.COMPOUND,
            ("animal", "group", "experiment", "study"): PadoColumn.STUDY,
        }
        organ = _collapse_dataframe_to_dict(odf, organ_structure, col_map)
        organ["finding"] = []
        for _, fdf in odf.groupby(col_map[PadoColumn.FINDING]):
            finding_structure = {
                (): PadoColumn.FINDING,
            }
            finding = _collapse_dataframe_to_dict(fdf, finding_structure, col_map)
            organ["finding"].append(finding)

        md["slide"]["organ"].append(organ)

    return md
