from __future__ import annotations

import warnings
from math import floor
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import StrictInt
from pydantic import conint
from pydantic.dataclasses import dataclass
from shapely.affinity import scale as shapely_scale
from shapely.geometry.base import BaseGeometry

__all__ = [
    "Point",
    "Size",
    "IntPoint",
    "IntSize",
    "MPP",
    "Bounds",
    "IntBounds",
    "Geometry",
    "ensure_type",
    "match_mpp",
]


def __getattr__(name):
    if name == "FuzzyMPP":
        warnings.warn(
            "MPP now supports setting atol and rtol for fuzzy matching. "
            "Please import `MPP` instead of `FuzzyMPP`",
            DeprecationWarning,
        )
        return MPP
    else:
        raise AttributeError(name)


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

    # support approximate matching
    rtol: NonNegativeFloat = 0.0  # relative tolerance
    atol: NonNegativeFloat = 0.0  # absolute tolerance

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

    def with_tolerance(self, rtol: float, atol: float) -> MPP:
        return MPP(self.x, self.y, rtol=rtol, atol=atol)

    @property
    def tolerance(self) -> Tuple[float, float]:
        tol_x = self.atol + self.rtol * abs(self.x)
        tol_y = self.atol + self.rtol * abs(self.y)
        return tol_x, tol_y

    @property
    def is_exact(self) -> bool:
        return self.rtol == 0 and self.atol == 0

    def _get_xy_tolerance(self, other: Any) -> Tuple[float, float, float, float]:
        if isinstance(other, MPP):
            atol = max(self.atol, other.atol)
            rtol_x = max(self.rtol, other.rtol * other.x / self.x)
            rtol_y = max(self.rtol, other.rtol * other.y / self.y)
            tol_x = atol + rtol_x * abs(self.x)
            tol_y = atol + rtol_y * abs(self.y)
            return other.x, other.y, tol_x, tol_y
        elif (
            isinstance(other, (tuple, list))
            and len(other) == 2
            and isinstance(other[0], (float, int))
            and isinstance(other[1], (float, int))
        ):
            tol_x, tol_y = self.tolerance
            return other[0], other[1], tol_x, tol_y
        else:
            raise TypeError(f"unsupported object of type: {type(other).__name__!r}")

    def __eq__(self, other: Any) -> bool:
        try:
            x, y, tol_x, tol_y = self._get_xy_tolerance(other)
        except TypeError:
            return False
        dx = abs(self.x - x)
        dy = abs(self.y - y)
        return dx <= tol_x and dy <= tol_y

    def __lt__(self, other: Any) -> bool:
        x, y, tol_x, tol_y = self._get_xy_tolerance(other)
        return self.x < (x - tol_x) and self.y < (y - tol_y)

    def __gt__(self, other: Any) -> bool:
        x, y, tol_x, tol_y = self._get_xy_tolerance(other)
        return self.x > (x + tol_x) and self.y > (y + tol_y)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)


def match_mpp(
    origin: MPP,
    *targets: MPP,
    remove_tolerance: bool = True,
    raise_no_match: bool = False,
) -> MPP:
    """returns an MPP from potential matches or the original"""
    targets = sorted(
        targets,
        key=lambda target: (origin.x - target.x) ** 2 + (origin.y - target.y) ** 2,
    )
    for t in targets:
        if origin == t:
            break
    else:
        if raise_no_match:
            raise ValueError("could not match to targets")
        t = origin
    if remove_tolerance:
        return t.with_tolerance(rtol=0, atol=0)
    else:
        return t


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
    def from_tuple(cls: Type[_S], xy: Tuple[float, float], *, mpp: Optional[MPP]) -> _S:
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
    """a class for rectangular bounds

    Notes
    -----

      p0 ---- +
       |      |
       |      |
       + ---- p1

    p0: Point(x0, y0)
    p1: Point(x1, y1)

    with: p0.x < p1.x and p0.y < p1.y

    """

    x0: NonNegativeFloat
    y0: NonNegativeFloat
    x1: NonNegativeFloat
    y1: NonNegativeFloat
    mpp: Optional[MPP] = None

    def __post_init_post_parse__(self):
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(
                f"Invalid bounds, must: {self.x0} < {self.x1} and {self.y0} < {self.y1}"
            )

    @property
    def x0y0(self) -> Point:
        return Point(self.x0, self.y0, mpp=self.mpp)

    @property
    def x1y1(self) -> Point:
        return Point(self.x0, self.y0, mpp=self.mpp)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def size(self) -> Size:
        return Size(x=self.width, y=self.height, mpp=self.mpp)

    def round(self) -> IntBounds:
        return IntBounds(
            round(self.x0),
            round(self.y0),
            round(self.x1),
            round(self.y1),
            self.mpp,
        )

    def floor(self) -> IntBounds:
        return IntBounds(
            floor(self.x0),
            floor(self.y0),
            floor(self.x1),
            floor(self.y1),
            self.mpp,
        )

    def scale(self, mpp: MPP) -> Bounds:
        """scale bounds to a new mpp"""
        current = self.mpp
        if current is None:
            raise ValueError(f"Can't scale: {self!r} has no mpp")
        return Bounds(
            x0=self.x0 * current.x / mpp.x,
            y0=self.y0 * current.y / mpp.y,
            x1=self.x1 * current.x / mpp.x,
            y1=self.y1 * current.y / mpp.y,
            mpp=mpp,
        )

    @classmethod
    def from_tuple(
        cls: Type[_B], x0y0x1y1: Tuple[float, float, float, float], *, mpp: MPP
    ) -> _B:
        return cls(*x0y0x1y1, mpp=mpp)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    def as_record(self) -> dict[str, float]:
        if self.mpp is None:
            raise ValueError("won't serialize without MPP")
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "mpp_x": self.mpp.x,
            "mpp_y": self.mpp.y,
        }

    @classmethod
    def from_record(cls, record) -> Bounds:
        return Bounds(
            record["x0"],
            record["y0"],
            record["x1"],
            record["y1"],
            mpp=MPP(
                record["mpp_x"],
                record["mpp_y"],
            ),
        )


# noinspection PyDataclass
@dataclass(frozen=True)
class IntBounds(Bounds):

    x0: conint(ge=0, strict=True)  # type: ignore
    y0: conint(ge=0, strict=True)  # type: ignore
    x1: conint(ge=0, strict=True)  # type: ignore
    y1: conint(ge=0, strict=True)  # type: ignore

    @property
    def x0y0(self) -> IntPoint:
        return IntPoint(self.x0, self.y0, mpp=self.mpp)

    @property
    def x1y1(self) -> IntPoint:
        return IntPoint(self.x0, self.y0, mpp=self.mpp)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def size(self) -> IntSize:
        return IntSize(x=self.width, y=self.height, mpp=self.mpp)

    def round(self) -> IntBounds:
        return self

    def floor(self) -> IntBounds:
        return self

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass(config=type("", (), {"arbitrary_types_allowed": True}))
class Geometry:
    """
    A general class for dealing with BaseGeometries at various MPPs.
    """

    geometry: BaseGeometry
    mpp: Optional[MPP] = None

    def scale(self, mpp: MPP) -> _G:
        """scale geometry to a new mpp"""
        current = self.mpp
        if current is None:
            raise ValueError(f"Can't scale: {self!r} has no mpp")

        factor_x = self.mpp.x / mpp.x
        factor_y = self.mpp.y / mpp.y

        return Geometry(
            geometry=shapely_scale(
                self.geometry, xfact=factor_x, yfact=factor_y, origin=(0, 0)
            ),
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


T = TypeVar("T")


def ensure_type(obj: Any, cls: type[T]) -> T:
    if isinstance(obj, cls):
        return obj
    elif isinstance(obj, dict):
        return cls(**obj)
    elif isinstance(obj, tuple):
        return cls(*obj)
    else:
        raise ValueError(f"could not cast {obj!r} to {cls.__name__}")
