"""tile classes for pado images"""
from __future__ import annotations

import inspect
import json
import math
import warnings
from itertools import islice
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
import orjson
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
from pado.images.utils import ensure_type
from pado.images.utils import match_mpp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pado.annotations import Annotations
    from pado.images.ids import ImageId
    from pado.images.image import Image

__all__ = [
    "FastGridTiling",
    "GridTileIndex",
    "PadoTileItem",
    "TileId",
    "TileIndex",
    "TilingStrategy",
]


def __getattr__(name):
    if name == "Tile":
        cls = _DeprecatedTile
    elif name == "TileIterator":
        cls = _DeprecatedTileIterator
    else:
        raise AttributeError(name)
    warnings.warn(
        f"`pado.images.tiles.{name}` will be removed in the next major version of pado."
        " Please checkout `pado.itertools.TileDataset`!",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls


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

    def precompute(
        self,
        image: Image,
        *,
        storage_options: dict[str, Any] | None = None,
    ) -> TileIndex:
        raise NotImplementedError

    def serialize(self) -> str:
        raise NotImplementedError

    @staticmethod
    def serialize_strategy_and_options(cls: type[TilingStrategy], **kwargs) -> str:
        name = cls.name
        if name is None:
            raise RuntimeError("cls.name must be set")
        kws = []
        for kw, kw_value in kwargs.items():
            v = json.dumps(kw_value, separators=(",", ":"))
            kws.append(f"{kw}={v}")
        return f"{name}:{';'.join(kws)}"

    @classmethod
    def parse_serialized_strategy_options(cls, strategy: str) -> dict[str, Any]:
        name, kwargs = strategy.split(":")
        if name != cls.name:
            raise RuntimeError(f"{cls!r} can't be used to parse: {strategy!r}")
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
    _registry = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        init_sig = inspect.signature(cls.__init__)
        for parameter in islice(init_sig.parameters.values(), 1, None):
            if parameter.kind not in {
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            }:
                raise ValueError(
                    f"subclass of TileIndex `{cls.__name__}`"
                    " must only use kw-only and **kwargs in __init__"
                )
        TileIndex._registry[cls.__name__] = cls

    def __getitem__(self, i: int) -> ReadTileTuple:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def to_json(self, *, as_string: bool = False) -> str | dict:
        cls_fqn = f"{inspect.getmodule(self).__name__}.{type(self).__name__}"
        init_sig = inspect.signature(self.__init__)
        if type(self) is TileIndex:
            raise RuntimeError(
                ".to_json() only makes sense for subclasses of TileIndex"
            )
        obj = {
            "type": "pado.images.tiles.TileIndex",
            "version": 1,
            "cls": cls_fqn,
            "kwargs": {
                name: getattr(self, f"_{name}")
                for name, parameter in init_sig.parameters.items()
                if parameter.kind == inspect.Parameter.KEYWORD_ONLY
            },
        }
        if as_string:
            return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode()
        else:
            return obj

    @classmethod
    def from_json(cls, obj: str | dict) -> TileIndex:
        if isinstance(obj, str):
            obj = orjson.loads(obj.encode())
        if not isinstance(obj, dict):
            raise TypeError("expected json str or dict")
        if obj["type"] != "pado.images.tiles.TileIndex":
            raise ValueError("not a tile index json")
        mod, cls_name = obj["cls"].rsplit(".", maxsplit=1)
        sub_cls = cls._registry[cls_name]
        return sub_cls(**obj["kwargs"])


class GridTileIndex(TileIndex):
    def __init__(
        self,
        *,
        image_size: IntSize,
        tile_size: IntSize,
        overlap: int,
        target_mpp: MPP,
        masked_indices: NDArray[np.int64] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        image_size = ensure_type(image_size, IntSize)
        tile_size = ensure_type(tile_size, IntSize)
        target_mpp = ensure_type(target_mpp, MPP)
        if tile_size.mpp is not None:
            if tile_size.mpp != target_mpp:
                raise NotImplementedError("tile_size.mpp must equal target_mpp")
            elif not tile_size.mpp.is_exact:
                raise ValueError("tile_size MPP must be exact!")

        if image_size.mpp is None:
            raise ValueError("image_size must provide an MPP!")
        elif not image_size.mpp.is_exact:
            raise ValueError("image_size MPP must be exact!")
        elif image_size.mpp != target_mpp:
            self._image_size = image_size.scale(target_mpp).round()
        else:
            self._image_size = image_size

        if not target_mpp.is_exact:
            raise ValueError("target_mpp MPP must be exact!")

        self._tile_size = tile_size
        self._overlap = int(overlap)
        if not (0 <= self._overlap < min(self._tile_size.x, self._tile_size.y)):
            raise ValueError(f"overlap is out of bounds: {self._overlap!r}")
        self._target_mpp = target_mpp
        if masked_indices is None:
            self._masked_indices = None
        else:
            self._masked_indices = np.array(masked_indices, dtype=np.int64, order="C")

    @classmethod
    def from_mask(
        cls,
        image_size: IntSize,
        tile_size: IntSize,
        overlap: int,
        target_mpp: MPP,
        mask: NDArray[np.bool] | None = None,
    ):
        if mask is None:
            masked_indices = None
        else:
            import cv2

            image_size, tile_size = cls._scale_size(image_size, tile_size, target_mpp)
            size = cls._get_size(image_size, tile_size, overlap)
            if not (mask.ndim == 2 and mask.dtype == bool):
                raise RuntimeError(
                    f"expected 2D boolean mask, got: {mask.shape!r} {mask.dtype!r}"
                )
            _mask = cv2.resize(
                mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            masked_indices = np.argwhere(_mask)
        return cls(
            image_size=image_size,
            tile_size=tile_size,
            overlap=overlap,
            target_mpp=target_mpp,
            masked_indices=masked_indices,
        )

    def __getitem__(self, item: int) -> tuple[IntPoint, IntSize, MPP]:
        item = int(item)
        sw, sh = self._image_size.as_tuple()
        tw, th = self._tile_size.as_tuple()
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

    @staticmethod
    def _get_size(
        image_size: IntSize, tile_size: IntSize, overlap: int
    ) -> tuple[int, int]:
        if image_size.mpp is None or image_size.mpp != tile_size.mpp:
            raise ValueError("image_size and tile_size must have same mpp")
        sw, sh = image_size.as_tuple()
        tw, th = tile_size.as_tuple()
        dx = tw - overlap
        dy = th - overlap
        num_x = math.floor(sw / dx)
        num_y = math.floor(sh / dy)
        return num_x, num_y

    @staticmethod
    def _scale_size(
        image_size: IntSize, tile_size: IntSize, target_mpp: MPP
    ) -> tuple[IntSize, IntSize]:
        if tile_size.mpp is not None:
            if tile_size.mpp != target_mpp:
                raise NotImplementedError("tile_size.mpp must equal target_mpp")

        if image_size.mpp is not None and image_size.mpp != target_mpp:
            return image_size.scale(target_mpp).round(), tile_size
        else:
            return image_size, tile_size

    def __len__(self):
        if self._masked_indices is not None:
            return len(self._masked_indices)
        else:
            num_x, num_y = self._get_size(
                self._image_size, self._tile_size, self._overlap
            )
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
        elif isinstance(target_mpp, MPP):
            self._target_mpp = target_mpp
        else:
            raise TypeError(
                f"target_mpp expected MPP | float, got {type(target_mpp).__name__!r}"
            )
        if not isinstance(tile_size, IntSize):
            self._tile_size = IntSize.from_tuple(tile_size, mpp=None)
        elif tile_size.mpp is None:
            self._tile_size = IntSize(tile_size.x, tile_size.y, mpp=None)
        elif tile_size.mpp != self._target_mpp:
            raise NotImplementedError("Todo: warn and scale?")
        else:
            self._tile_size = IntSize(tile_size.x, tile_size.y, mpp=None)
        self._overlap = int(overlap)
        self._min_chunk_size = min_chunk_size
        self._normalize_chunk_size = normalize_chunk_sizes

    def precompute(
        self,
        image: Image,
        *,
        storage_options: dict[str, Any] | None = None,
    ) -> TileIndex:
        image_size = IntSize(
            image.metadata.width,
            image.metadata.height,
            mpp=MPP(image.metadata.mpp_x, image.metadata.mpp_y),
        )
        target_mpp = match_mpp(self._target_mpp, *image.level_mpp.values())

        if self._min_chunk_size is not None:
            with image.open(storage_options=storage_options):
                chunk_sizes = image.get_chunk_sizes(level=0)
            if self._normalize_chunk_size:
                if np.min(chunk_sizes) == np.max(chunk_sizes):
                    warnings.warn(f"all chunksizes identical: {image!r}")
                chunk_sizes = (chunk_sizes - np.min(chunk_sizes)) / np.max(chunk_sizes)
            mask = chunk_sizes >= self._min_chunk_size
        else:
            mask = None

        if not target_mpp.is_exact:
            raise RuntimeError("target_mpp must be exact")
        if self._tile_size.mpp is not None:
            raise RuntimeError("tile_size.mpp can't be set before")
        else:
            tile_size = IntSize(self._tile_size.x, self._tile_size.y, mpp=target_mpp)

        return GridTileIndex.from_mask(
            image_size=image_size,
            tile_size=tile_size,
            overlap=self._overlap,
            target_mpp=target_mpp,
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


# === obsolete ========================================================


class _DeprecatedTile:
    """pado.img.Tile abstracts rectangular regions in whole slide image data"""

    def __init__(
        self,
        mpp: MPP,
        lvl0_mpp: MPP,
        bounds: Bounds,
        data: Optional[np.ndarray] = None,
        parent: Optional[Image] = None,
    ):
        if mpp.as_tuple() != bounds.mpp.as_tuple():
            raise NotImplementedError(
                f"tile mpp does not coincide with bounds mpp: {mpp} vs {bounds.mpp}"
            )
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


class _DeprecatedTileIterator:
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

    def __iter__(self) -> Iterator[_DeprecatedTile]:
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
                    yield _DeprecatedTile(
                        mpp=mpp_xy,
                        lvl0_mpp=self.level0_mpp_xy,
                        bounds=Bounds.from_tuple((x0, x1, y0, y1), mpp=mpp_xy),
                        data=z_array[y0:y1, x0:x1],
                        parent=img,
                    )

        yield from _yield_tiles(store)
