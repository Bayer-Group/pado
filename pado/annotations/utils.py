from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from shapely.affinity import scale as shapely_scale
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry

from pado.images.utils import MPP


__all__ = [
    "Geometry"
]


_G = TypeVar("_G", bound="Geometry")


@dataclass(frozen=True)
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