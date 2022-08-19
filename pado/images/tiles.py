"""tile classes for pado images"""
from __future__ import annotations

import json
import math
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
import zarr
from shapely.geometry import Polygon
from typing_extensions import TypeAlias

from pado._compat import cached_property
from pado.images.utils import MPP
from pado.images.utils import Bounds
from pado.images.utils import Geometry
from pado.images.utils import IntPoint
from pado.images.utils import IntSize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pado.annotations import Annotations
    from pado.images.ids import ImageId
    from pado.images.image import Image


class TileId(NamedTuple):
    """a predictable tile id"""

    image_id: ImageId
    strategy: str
    index: int


class PadoTileItem(NamedTuple):
    """A 'row' of a dataset of tiles generated from a PadoDataset"""

    id: TileId
    tile: NDArray
    annotations: Optional[Annotations]
    metadata: Optional[pd.DataFrame]


class TilingStrategy:
    name: str | None = None

    def precompute(self, image: Image) -> TileIndex:
        raise NotImplementedError

    def serialize(self) -> str:
        raise NotImplementedError

    @staticmethod
    def serialize_strategy_and_options(cls: type[TilingStrategy], **kwargs) -> str:
        name = cls.name
        assert name is not None
        kws = []
        for kw, kw_value in kwargs.items():
            v = json.dumps(kw_value, separators=(",", ":"))
            kws.append(f"{kw}={v}")
        return f"{name}:{';'.join(kws)}"

    @classmethod
    def parse_serialized_strategy_options(cls, strategy: str) -> dict[str, Any]:
        name, kwargs = strategy.split(":")
        assert name == cls.name
        kws = {}
        for kw_val in kwargs.split(";"):
            key, val = kw_val.split("=")
            kws[key] = json.loads(val)

        return kws

    @classmethod
    def deserialize(cls, strategy: str) -> TilingStrategy:
        if not isinstance(strategy, str):
            raise TypeError(f"expected str, got {type(strategy).__name__}")
        name, kwargs = strategy.split(":")
        for s_cls in cls.__subclasses__():
            if s_cls.name == name:
                break
        else:
            raise ValueError(f"could not find matching strategy: {strategy!r}")
        s_cls, kwargs = cls.parse_serialized_strategy_options(strategy)
        return s_cls(**kwargs)


ReadTileTuple: TypeAlias = "tuple[IntPoint, IntSize, MPP]"


class TileIndex(Sequence[ReadTileTuple]):
    def __getitem__(self, i: int) -> ReadTileTuple:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class GridTileIndex(TileIndex):
    def __init__(
        self,
        image_size: IntSize,
        tile_size: IntSize,
        overlap: int,
        target_mpp: MPP,
        mask: NDArray[np.bool] | None = None,
    ):
        if tile_size.mpp is not None:
            assert tile_size.mpp == target_mpp

        if image_size.mpp is not None and image_size.mpp != target_mpp:
            self._image_size = image_size.scale(target_mpp).as_tuple()
        else:
            self._image_size = image_size.as_tuple()

        self._tile_size = tile_size
        self._overlap = int(overlap)
        assert 0 <= self._overlap < min(self._tile_size.x, self._tile_size.y)
        self._target_mpp = target_mpp

        if mask is None:
            self._masked_indices = None
        else:
            import cv2

            size = self._get_size()
            assert mask.ndim == 2 and mask.dtype == bool
            _mask = cv2.resize(
                mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            self._masked_indices = np.argwhere(_mask)

    def __getitem__(self, item: int) -> tuple[IntPoint, IntSize, MPP]:
        item = int(item)
        sw, sh = self._image_size
        tw, th = self._tile_size.x, self._tile_size.y
        dx = tw - self._overlap
        dy = th - self._overlap
        num_x = math.floor(sw / dx)
        num_y = math.floor(sh / dy)
        num = num_x * num_y

        if self._masked_indices is None:
            if 0 <= item < num:
                pass
            elif -num <= item < 0:
                item += num
            else:
                raise IndexError(item)
            x = item % num_x
            y = item // num_x
        else:
            y, x = map(int, self._masked_indices[item])

        return (
            IntPoint(x * dx, y * dy, mpp=self._target_mpp),
            self._tile_size,
            self._target_mpp,
        )

    def _get_size(self):
        sw, sh = self._image_size
        tw, th = self._tile_size.x, self._tile_size.y
        dx = tw - self._overlap
        dy = th - self._overlap
        num_x = math.floor(sw / dx)
        num_y = math.floor(sh / dy)
        return num_x, num_y

    def __len__(self):
        if self._masked_indices is not None:
            return len(self._masked_indices)
        else:
            num_x, num_y = self._get_size()
            return num_x * num_y


class FastGridTiling(TilingStrategy):
    name = "fastgrid"

    def __init__(
        self,
        *,
        tile_size: IntSize | tuple[int, int],
        target_mpp: MPP | float,
        overlap: int = 0,
        min_chunk_size: float | int | None,
        normalize_chunk_sizes: bool,
    ) -> None:
        if isinstance(target_mpp, float):
            self._target_mpp = MPP.from_float(target_mpp)
        elif not isinstance(target_mpp, MPP):
            raise TypeError(
                f"target_mpp expected MPP | float, got {type(target_mpp).__name__!r}"
            )
        else:
            self._target_mpp = target_mpp
        if isinstance(tile_size, IntSize):
            if tile_size.mpp is None:
                self._tile_size = IntSize(
                    tile_size.x, tile_size.y, mpp=self._target_mpp
                )
            elif tile_size.mpp != self._target_mpp:
                raise NotImplementedError("Todo: warn and scale?")
            else:
                self._tile_size = tile_size
        else:
            tw, th = tile_size
            self._tile_size = IntSize(tw, th, mpp=self._target_mpp)
        self._overlap = int(overlap)
        self._min_chunk_size = min_chunk_size
        self._normalize_chunk_size = normalize_chunk_sizes

    def precompute(self, image: Image) -> TileIndex:
        image_size = IntSize(
            image.metadata.width,
            image.metadata.height,
            mpp=MPP(image.metadata.mpp_x, image.metadata.mpp_y),
        )
        if self._min_chunk_size is not None:
            with image:
                chunk_sizes = image.get_chunk_sizes(level=0)
            if self._normalize_chunk_size:
                if np.min(chunk_sizes) == np.max(chunk_sizes):
                    warnings.warn("all chunksizes identical: {image!r}")
                chunk_sizes = (chunk_sizes - np.min(chunk_sizes)) / np.max(chunk_sizes)
            mask = chunk_sizes >= self._min_chunk_size
        else:
            mask = None

        return GridTileIndex(
            image_size=image_size,
            tile_size=self._tile_size,
            overlap=self._overlap,
            target_mpp=self._target_mpp,
            mask=mask,
        )

    def serialize(self) -> str:
        return self.serialize_strategy_and_options(
            type(self),
            tile_size=(self._tile_size.x, self._tile_size.y),
            target_mpp=(self._target_mpp.x, self._target_mpp.y),
            overlap=self._overlap,
            min_chunk_size=self._min_chunk_size,
            normalize_chunk_size=self._normalize_chunk_size,
        )


# === potentially obsolete =====================================


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
        return self.bounds.round().size

    @cached_property
    def x0y0(self) -> IntPoint:
        return self.bounds.round().x0y0

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
