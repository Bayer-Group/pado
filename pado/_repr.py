"""useful repr tools for everything pado dataset related"""
from __future__ import annotations

import math
import sys
import textwrap
from enum import Enum
from numbers import Number
from reprlib import Repr
from typing import Any
from typing import Callable
from typing import Iterable
from typing import cast

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import pandas as pd

__all__ = [
    "mapping_repr",
    "DescribeFormat",
    "describe_format_plain_text",
    "number",
    "is_number",
    "number_to_str",
    "is_mpp_count",
]

# === common repr functions ===========================================

_r = Repr()
_r.maxlist = 3


def mapping_repr(inst: Any) -> str:
    _akw = [_r.repr_dict(cast(dict, inst), 0)]
    if hasattr(inst, "identifier") and inst.identifier is not None:
        _akw.append(f"identifier={inst.identifier!r}")
    return f"{type(inst).__name__}({', '.join(_akw)})"


# === dataset description =============================================


class DescribeFormat(str, Enum):
    """supported formats for PadoDataset.describe"""

    PLAIN_TEXT = "plain_text"
    DICT = "dict"
    JSON = "json"


def describe_format_plain_text(data: dict) -> str:
    """return a string representation of the dataset description"""

    def _disp(fun, dct, ind):
        return textwrap.indent(
            "\n".join(map(fun, dct)),
            "  " * ind,
        ).strip()

    return textwrap.dedent(
        """\
        ## SUMMARY ##
        path: {path}
        images: {num_images}

        ### IMAGES ###
        image_information:
          - mpps:
            {disp_mpps}
          - width: {disp_image_width}
          - height: {disp_image_height}
        file_information:
          - size: {disp_file_size}
          - total_size: {total_size_images}

        ### ANNOTATIONS ###
        annotations:
          - total: {total_num_annotations}
          - per_image: {disp_annotations_per_image}
        classes:
          - most_common_number:
            {disp_common_classes_num}
          - most_common_area:
            {disp_common_classes_area}

        ### METADATA ###
        columns:
          {disp_metadata_columns}
    """
    ).format(
        path=repr(data["path"]),
        num_images=data["num_images"],
        disp_image_width=number_to_str(data["avg_image_width"]),
        disp_image_height=number_to_str(data["avg_image_height"]),
        disp_file_size=number_to_str(data["avg_image_size"], prefix_unit=True),
        total_size_images=number_to_str(data["total_size_images"], prefix_unit=True),
        total_num_annotations=data["total_num_annotations"],
        disp_annotations_per_image=number_to_str(data["avg_annotations_per_image"]),
        disp_mpps=_disp(lambda d: f"- {d['mpp']!r}: {d['num']}", data["num_mpps"], 2),
        disp_common_classes_num=_disp(
            lambda t: f"- {t[0]!r}: {t[1]}", data["common_classes_count"].items(), 2
        ),
        disp_common_classes_area=_disp(
            lambda t: f"- {t[0]!r}: {number_to_str(t[1], prefix='~', prefix_unit=True)}",
            data["common_classes_area"].items(),
            2,
        ),
        disp_metadata_columns=_disp(lambda c: f"- {c}", data["metadata_columns"], 1),
    )


# === number formatting functions =====================================


def number(
    v: Iterable[Number] | Number,
    name: str | None = None,
    agg: Literal["sum", "avg", "id"] = "id",
    cast_to: Callable[[Any], Any] = int,
    unit: str | None = None,
) -> dict[str, Any] | Number:
    """create a number with some meta information"""
    if isinstance(v, Number):
        v = pd.Series([v])
    elif isinstance(v, pd.DataFrame):
        v = v[name]
    elif isinstance(v, pd.Series):
        pass
    else:
        raise ValueError(f"{v!r}")

    x = {}
    if agg == "id":
        x["val"] = cast_to(v.item())
    elif agg == "sum":
        x["val"] = cast_to(v.sum())  # type: ignore
    elif agg == "avg":
        x["val"] = cast_to(v.mean())
        x["std"] = cast_to(v.std())
    else:
        raise ValueError(f"agg={agg!r}")

    if unit:
        x["unit"] = unit
    elif len(x) == 1:
        return x["val"]

    return x


def is_number(obj: Any) -> bool:
    """check if a number"""
    if isinstance(obj, Number):
        return True
    elif isinstance(obj, dict):
        return "val" in obj and {"val", "std", "unit"}.issuperset(obj)
    else:
        return False


def number_to_str(
    v: dict | Number,
    *,
    prefix: str = "",
    prefix_unit: bool = False,
) -> str:
    """transform numbers to a string representation

    Note: this is not complete at all, and only implements what we need right now...
    """
    if isinstance(v, dict):
        val = v["val"]
        std = v.get("std")
        unit = v.get("unit", "")
    else:
        val = v
        std = None
        unit = ""

    r = max(0, int(math.log10((std if std is not None else val) + 0.1)) - 1)
    u = (r // 3) if prefix_unit else 0

    val = round(val, -r)
    val /= 10 ** (u * 3)
    unit = f'{("", "k", "M", "G", "T", "P")[u]}{unit}'  # todo: more?

    if std is not None:
        std = round(std, -r)
        std /= 10 ** (u * 3)
        return f"{prefix}{int(val)} +- {int(std)} {unit}".strip()
    else:
        return f"{prefix}{int(val)} {unit}".strip()


def is_mpp_count(obj: Any) -> bool:
    """check if a number"""
    return isinstance(obj, dict) and {"mpp", "num"} == set(obj)
