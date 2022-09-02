from __future__ import annotations

import copy
import warnings
from reprlib import Repr
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import MutableSequence
from typing import Optional
from typing import Union
from typing import overload

import orjson
import pandas as pd
from pydantic.color import Color
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.wkt import loads as wkt_loads

from pado.annotations.formats import AnnotationModel
from pado.annotations.formats import AnnotationState
from pado.annotations.formats import AnnotationStyle
from pado.annotations.formats import Annotator
from pado.images.ids import ImageId

if TYPE_CHECKING:
    from pado.images.utils import MPP
    from pado.images.utils import IntPoint


class Annotation:
    """Annotation class"""

    image_id: Optional[ImageId]
    identifier: Optional[str]
    project: Optional[str]
    annotator: Annotator
    state: AnnotationState
    style: AnnotationStyle
    classification: str
    color: Color
    description: str
    comment: str
    geometry: BaseGeometry

    def __init__(self, model: AnnotationModel):
        # noinspection PyProtectedMember
        self.__dict__.update(model._iter())
        self._model = model
        self._readonly = True

    def __setattr__(self, key, value):
        if getattr(self, "_readonly", False):
            raise AttributeError(f"{key} is readonly")
        super().__setattr__(key, value)

    def __repr__(self):
        return f"{type(self).__name__}(model={self._model!r})"

    def __eq__(self, other):
        if not isinstance(other, Annotation):
            return False
        return all(
            self.__dict__[k] == other.__dict__[k]
            for k in self.__dict__
            if k not in {"_model", "color"}
        )

    @classmethod
    def from_obj(cls, obj: Any) -> Annotation:
        """instantiate an annotation from an object, i.e. a pd.Series"""
        return cls(AnnotationModel.parse_obj(obj))

    def to_record(self, image_id: Optional[ImageId] = None) -> dict:
        """return a record for serializing"""
        m = self._model
        dct = m.dict(exclude={"image_id", "color", "geometry"})

        if m.image_id is not None and image_id is not None:
            if m.image_id != image_id:
                raise ValueError(
                    f"Annotation has different image_id: has {m.image_id} requested {image_id}"
                )

        _id = m.image_id or image_id
        dct["image_id"] = _id.to_str() if _id is not None else None
        dct["color"] = m.color.as_rgb() if m.color is not None else None
        dct["geometry"] = m.geometry.wkt
        return dct


_r = Repr()
_r.maxlist = 3


class Annotations(MutableSequence[Annotation]):
    df: pd.DataFrame

    def __init__(
        self, df: Optional[pd.DataFrame] = None, *, image_id: Optional[ImageId] = None
    ) -> None:
        if df is None:
            self.df = pd.DataFrame(columns=AnnotationModel.__fields__)
        elif isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError(f"requires a pd.DataFrame, not {type(df).__name__}")
        self._image_id = image_id
        if image_id is not None:
            self._update_df_image_id(image_id)

    def __repr__(self):
        return f"{type(self).__name__}({_r.repr_list(self, 0)}, image_id={self._image_id!r})"

    def __eq__(self, other):
        if not isinstance(other, Annotations):
            return False
        return all(a == b for a, b in zip(self, other))

    @property
    def image_id(self) -> Optional[ImageId]:
        return self._image_id

    def _update_df_image_id(self, image_id: ImageId):
        """internal"""
        if self.df.empty:
            return
        ids = set(self.df["image_id"].unique())
        if len(ids) > 2:
            raise ValueError(f"image_ids in provider not unique: {ids!r}")
        if None not in ids and image_id.to_str() in ids:
            return
        elif {None, image_id.to_str()}.issuperset(ids):
            self.df.loc[self.df["image_id"].isna(), "image_id"] = image_id.to_str()
        else:
            raise AssertionError(f"unexpected image_ids in Annotations.df: {ids!r}")

    @image_id.setter
    def image_id(self, value: ImageId):
        if not isinstance(value, ImageId):
            raise TypeError(
                f"{value!r} not of type ImageId, got {type(value).__name__}"
            )
        self._update_df_image_id(image_id=value)
        self._image_id = value

    @overload
    def __getitem__(self, index: int) -> Annotation:
        ...

    @overload
    def __getitem__(self, index: slice) -> Annotations:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Annotation, Annotations]:
        if isinstance(index, int):
            return Annotation.from_obj(self.df.iloc[index, :])
        elif isinstance(index, slice):
            return Annotations(self.df.loc[index, :], image_id=self.image_id)
        else:
            raise TypeError(
                f"Annotations: indices must be integers or slices, not {type(index).__name__}"
            )

    @overload
    def __setitem__(self, index: int, value: Annotation) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Annotation]) -> None:
        ...

    def __setitem__(
        self, index: Union[int, slice], value: Union[Annotation, Iterable[Annotation]]
    ) -> None:
        if isinstance(index, int):
            self.df.iloc[index, :] = pd.DataFrame(
                [value.to_record(self._image_id)], columns=AnnotationModel.__fields__
            )
        elif isinstance(index, slice):
            self.df.iloc[index, :] = pd.DataFrame(
                [x.to_record(self._image_id) for x in value],
                columns=AnnotationModel.__fields__,
            )
        else:
            raise TypeError(
                f"Annotations: indices must be integers or slices, not {type(index).__name__}"
            )

    def __delitem__(self, index: Union[int, slice]) -> None:
        if isinstance(index, int):
            self.df.drop(labels=index, axis=0, inplace=True)
        elif isinstance(index, slice):
            self.df.drop(labels=self.df.index[index], axis=0, inplace=True)
        else:
            raise TypeError(
                f"Annotations: indices must be integers or slices, not {type(index).__name__}"
            )

    def insert(self, index: int, value: Annotation) -> None:
        if not isinstance(value, Annotation):
            raise TypeError(
                f"can only insert type Annotation, got {type(value).__name__!r}"
            )
        df_a = self.df.iloc[:index, :]
        df_i = pd.DataFrame(
            [value.to_record(self._image_id)], columns=AnnotationModel.__fields__
        )
        df_b = self.df.iloc[index:, :]
        self.df = pd.concat([df_a, df_i, df_b])

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_records(
        cls, annotation_records: Iterable[dict], *, image_id: Optional[ImageId] = None
    ) -> Annotations:
        df = pd.DataFrame(list(annotation_records), columns=AnnotationModel.__fields__)
        return Annotations(df, image_id=image_id)


class AnnotationIndex:
    def __init__(self, geometries: list[BaseGeometry]) -> None:
        self.geometries = copy.copy(geometries)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            strtree = STRtree(geometries)
        self._strtree = strtree

    # noinspection PyShadowingNames
    @classmethod
    def from_annotations(
        cls, annotations: Annotations | None
    ) -> AnnotationIndex | None:
        if annotations is None:
            return None
        geometries = [a.geometry for a in annotations]
        return cls(geometries)

    def query_items(self, geom: BaseGeometry) -> list[int]:
        return list(self._strtree.query_items(geom))

    def to_json(self, *, as_string: bool = False) -> str | dict:
        obj = {
            "type": "pado.annotations.annotation.AnnotationIndex",
            "version": 1,
            "geometries": [o.wkt for o in self.geometries],
        }
        if as_string:
            return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode()
        else:
            return obj

    @classmethod
    def from_json(cls, obj: str | dict | None) -> AnnotationIndex | None:
        if obj is None:
            return None
        if isinstance(obj, str):
            obj = orjson.loads(obj.encode())
        if not isinstance(obj, dict):
            raise TypeError("expected json str or dict")

        t = obj["type"]
        if t != "pado.annotations.annotation.AnnotationIndex":
            raise NotImplementedError(t)
        geometries = obj["geometries"]
        return cls([wkt_loads(o) for o in geometries])


def shapely_fix_shape(
    shape: BaseGeometry, buffer_size: tuple[int, int]
) -> BaseGeometry:
    shape = shape.buffer(buffer_size[0])
    if not shape.is_valid:
        shape = shape.buffer(buffer_size[1])
    return shape


def ensure_validity(
    annotation: Annotation,
) -> Annotation:
    geom = annotation.geometry
    if not geom.is_valid:
        geom = shapely_fix_shape(geom, buffer_size=(0, 0))
    annotation.geometry = geom
    return annotation


def scale_annotation(
    annotation: Annotation,
    *,
    level0_mpp: MPP,
    target_mpp: MPP,
) -> Annotation:
    rescale = None
    # We rescale if target_mpp differs from slide_mpp
    if target_mpp != level0_mpp:
        rescale = dict(
            xfact=level0_mpp.x / target_mpp.x,
            yfact=level0_mpp.y / target_mpp.y,
            origin=(0, 0),
        )

    geom = annotation.geometry
    if not geom.is_valid:
        geom = shapely_fix_shape(geom, buffer_size=(0, 0))

    if rescale:
        geom = shapely_scale(geom, **rescale)

    if not geom.is_valid:
        geom = shapely_fix_shape(geom, buffer_size=(0, 0))

    annotation.geometry = geom
    return annotation


def translate_annotation(annotation: Annotation, *, location: IntPoint) -> Annotation:
    geom = shapely_translate(annotation.geometry, xoff=-location.x, yoff=-location.y)
    annotation.geometry = geom
    return annotation
