from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

from pydantic import PositiveFloat
from pydantic import StrictInt
from pydantic import conint
from pydantic.dataclasses import dataclass
from shapely.affinity import scale as shapely_scale

__all__ = [
    "Point",
    "Size",
    "IntPoint",
    "IntSize",
    "MPP",
    "Bounds",
    "Geometry"
]

# NOTE:
#  maybe we should use decimals instead of floats here, to be really correct.
#  Why you ask? There's some strange edge cases that we should further investigate,
#  where downsample levels in svs files are slightly off due to integer sizes
#  of the pyramidal layers.
#  Anyways, this is just a reminder in case we run into problems in the future.
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class MPP:
    """micrometer per pixel scaling common in pathological images"""
    x: PositiveFloat
    y: PositiveFloat

    def scale(self, downsample: float) -> MPP:
        return MPP(x=self.x * downsample, y=self.y * downsample)

    @classmethod
    def from_float(cls, xy: float) -> MPP:
        return MPP(x=xy, y=xy)

    @classmethod
    def from_tuple(cls, xy: Tuple[float, float]) -> MPP:
        x, y = xy
        return MPP(x=x, y=y)

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


_P = TypeVar("_P", bound="Point")
_S = TypeVar("_S", bound="Size")
_B = TypeVar("_B", bound="Bounds")
_G = TypeVar("_G", bound="Geometry")


@dataclass(frozen=True)
class Point:
    """a 2D point that can optionally come with a MPP for scaling"""
    x: float
    y: float
    mpp: Optional[MPP] = None

    def round(self, method: Callable[[float], int] = round) -> IntPoint:
        return IntPoint(method(self.x), method(self.y), self.mpp)

    def scale(self, mpp: MPP) -> Point:
        """scale a point to a new mpp"""
        current = self.mpp
        if current is None:
            raise ValueError(f"Can't scale: {self!r} has no mpp")
        return Point(
            x=self.x * current.x / mpp.x,
            y=self.y * current.y / mpp.y,
            mpp=mpp,
        )

    @classmethod
    def from_tuple(cls: Type[_P], xy: Tuple[float, float], *, mpp: MPP) -> _P:
        x, y = xy
        return cls(x=x, y=y, mpp=mpp)

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


# noinspection PyDataclass
@dataclass(frozen=True)
class IntPoint(Point):
    """an integer 2D point"""
    x: StrictInt
    y: StrictInt

    def round(self, method=None) -> IntPoint:
        return self

    def as_tuple(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class Size:
    """a general 2D size that can optionally come with a MPP for scaling"""
    x: PositiveFloat
    y: PositiveFloat
    mpp: Optional[MPP] = None

    def round(self) -> IntSize:
        return IntSize(round(self.x), round(self.y), self.mpp)

    def scale(self, mpp: MPP) -> Size:
        """scale a size to a new mpp"""
        current = self.mpp
        if current is None:
            raise ValueError(f"Can't scale: {self!r} has no mpp")
        return Size(
            x=self.x * current.x / mpp.x,
            y=self.y * current.y / mpp.y,
            mpp=mpp,
        )

    @property
    def width(self):
        return self.x

    @property
    def height(self):
        return self.y

    @classmethod
    def from_tuple(cls: Type[_S], xy: Tuple[float, float], *, mpp: MPP) -> _S:
        x, y = xy
        return cls(x=x, y=y, mpp=mpp)

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


# noinspection PyDataclass
@dataclass(frozen=True)
class IntSize(Size):
    """an integer 2D size"""
    x: conint(gt=0, strict=True)  # type: ignore
    y: conint(gt=0, strict=True)  # type: ignore

    def round(self) -> IntSize:
        return self

    def as_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Bounds:
    """
    A general 4D size that aims at representing rectangular shapes.
    It optionally comes with a MPP for scaling
    """
    x_left: PositiveFloat
    y_left: PositiveFloat
    x_right: PositiveFloat
    y_right: PositiveFloat
    mpp: Optional[MPP] = None

    def round(self) -> Bounds:
        return Bounds(round(self.x_left), round(self.y_left), round(self.x_right), round(self.y_right), self.mpp)

    def scale(self, mpp: MPP) -> Bounds:
        """scale bounds to a new mpp"""
        current = self.mpp
        if current is None:
            raise ValueError(f"Can't scale: {self!r} has no mpp")
        return Bounds(
            x_left=self.x_left * current.x / mpp.x,
            y_left=self.y_left * current.y / mpp.y,
            x_right=self.x_right * current.x / mpp.x,
            y_right=self.y_right * current.y / mpp.y,
            mpp=mpp,
        )

    @property
    def upper_left_coords(self) -> Size:
        return Size(x=self.x_left, y=self.y_left, mpp=self.mpp)

    @property
    def width(self):
        return self.x_right - self.x_left

    @property
    def height(self):
        return self.y_right - self.y_left

    @property
    def size(self) -> IntSize:
        return IntSize(x=self.width, y=self.height, mpp=self.mpp)

    @classmethod
    def from_tuple(cls: Type[_B], xyxy: Tuple[float, float, float, float], *, mpp: MPP) -> _B:
        x_left, y_left, x_right, y_right = xyxy
        return cls(x_left=x_left, y_left=y_left, x_right=x_right, y_right=y_right, mpp=mpp)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x_left, self.y_left, self.x_right, self.y_right


@dataclass(frozen=True)
class Geometry:
    """
    A general class for dealing with BaseGeometries at various MPPs.
    """
    geometry: Optional[BaseGeometry]
    mpp: Optional[MPP] = None

    def scale(self, mpp: MPP) -> _G:
        """scale geometry to a new mpp"""
        current = self.mpp
        if current is None:
            raise ValueError(f"Can't scale: {self!r} has no mpp")

        factor_x = self.mpp.x / mpp.x
        factor_y = self.mpp.y / mpp.y

        return Geometry(
            geometry=shapely_scale(self.geometry, xfact=factor_x, yfact=factor_y, origin=(0, 0)),
            mpp=mpp,
        )

    @property
    def is_valid(self):
        return self.geometry.is_valid

    def fix_geometry(self, buffer_size: Optional[Tuple[int, int]] = None):
        if buffer_size is None:
            buffer_size = (0, 0)
        if not self.is_valid:
            self.geometry = self.geometry.buffer(buffer_size[0])
            if not self.is_valid:
                self.geometry = self.geometry.buffer(buffer_size[0])

    @classmethod
    def from_geometry(cls: Type[_G], geometry: BaseGeometry, *, mpp: MPP) -> _G:
        return cls(geometry=geometry, mpp=mpp)
