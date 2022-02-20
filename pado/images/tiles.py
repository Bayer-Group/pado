"""tile classes for pado images"""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
from typing import Optional

import numpy as np
import zarr
from shapely.geometry import Polygon

from pado._compat import cached_property
from pado.images.utils import MPP
from pado.images.utils import Bounds
from pado.images.utils import Geometry
from pado.images.utils import IntPoint
from pado.images.utils import IntSize

if TYPE_CHECKING:
    from pado.images.image import Image


class Tile:
    """pado.img.Tile abstracts rectangular regions in whole slide image data"""

    def __init__(
        self,
        mpp: MPP,
        lvl0_mpp: MPP,
        bounds: Bounds,
        data: Optional[np.ndarray] = None,
        parent: Optional[Image] = None,
    ):
        assert (
            mpp.as_tuple() == bounds.mpp.as_tuple()
        ), f"tile mpp does not coincide with bounds mpp: {mpp} vs {bounds.mpp}"
        self.mpp = mpp
        self.level0_mpp = lvl0_mpp
        self.bounds = bounds
        self.data: Optional[np.ndarray] = data
        self.parent: Optional[Image] = parent

        # compute quantities at level0. This is useful when, for instance, visualizing objects on original svs
        self.level0_bounds = bounds.scale(mpp=self.level0_mpp)
        self.level0_tile_size = self.size.scale(mpp=self.level0_mpp)
        self.level0_x0y0 = self.x0y0.scale(mpp=self.level0_mpp)

    @cached_property
    def size(self) -> IntSize:
        return self.bounds.size

    @cached_property
    def x0y0(self) -> IntPoint:
        return IntPoint.from_tuple(
            self.bounds.upper_left_coords.as_tuple(), mpp=self.mpp
        )

    def shape(self, mpp: Optional[MPP] = None) -> Geometry:
        if mpp is None:
            return Geometry.from_geometry(
                geometry=Polygon.from_bounds(*self.bounds.as_tuple()), mpp=self.mpp
            )
        else:
            return Geometry.from_geometry(
                geometry=Polygon.from_bounds(*self.bounds.scale(mpp=mpp).as_tuple()),
                mpp=mpp,
            )


class TileIterator:
    """helper class to iterate over tiles

    Note: we should subclass to enable all sorts of fancy tile iteration

    """

    def __init__(
        self,
        image: Image,
        *,
        size: IntSize,
        level: int,
    ):
        """create a tile iterator instance"""
        if not isinstance(image, Image):
            raise TypeError(
                f"expected Image, got {image!r} of type {type(image).__name__}"
            )
        if not isinstance(size, IntSize):
            raise TypeError(
                f"expected IntSize, got {size!r} of type {type(size).__name__}"
            )
        if not 0 <= int(level) < image.level_count:
            raise ValueError(
                "level={self.level} not in range({self.image.level_count})"
            )
        self.image: Image = image
        self.size: IntSize = size
        self.level: int = int(level)

        with self.image:
            self.level0_mpp_xy = self.image.level_mpp[0]

    def __iter__(self) -> Iterator[Tile]:
        """return a plain iterator with no overlap over all tiles of the image

        Note: boundary tiles that don't meet the size requirements are discarded
        """
        img_lvl = self.image.level_dimensions[self.level]
        tile_size = self.size
        img = self.image

        # todo: incomplete tiles at borders are currently discarded
        x, y = np.mgrid[
            0 : img_lvl.width - tile_size.width + 1 : tile_size.width,
            0 : img_lvl.height - tile_size.height + 1 : tile_size.height,
        ]

        # todo: check if this ordering makes sense? maybe depend on chunk order in zarr
        bounds = np.hstack(
            (
                x.reshape(-1, 1),
                x.reshape(-1, 1) + tile_size.width,
                y.reshape(-1, 1),
                y.reshape(-1, 1) + tile_size.height,
            )
        )

        mpp_xy = self.image.level_mpp[self.level]
        store = self.image.get_zarr_store(self.level)

        def _yield_tiles(s):
            with s:
                z_array = zarr.open_array(s, mode="r")
                for x0, x1, y0, y1 in bounds:
                    yield Tile(
                        mpp=mpp_xy,
                        lvl0_mpp=self.level0_mpp_xy,
                        bounds=Bounds.from_tuple((x0, x1, y0, y1), mpp=mpp_xy),
                        data=z_array[y0:y1, x0:x1],
                        parent=img,
                    )

        yield from _yield_tiles(store)
