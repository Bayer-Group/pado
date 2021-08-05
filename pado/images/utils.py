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

__all__ = [
    "Point",
    "Size",
    "IntPoint",
    "IntSize",
    "MPP",
]

# NOTE:
#  maybe we should use decimals instead of floats here, to be really correct.
#  Why you ask? There's some strange edge cases that we should further investigate,
#  where downsample levels in svs files are slightly off due to integer sizes
#  of the pyramidal layers.
#  Anyways, this is just a reminder in case we run into problems in the future.


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
        return self.x, self.y
