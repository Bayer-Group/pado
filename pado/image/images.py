"""pado image abstraction to hide image loading implementation"""
from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple

import fsspec
import fsspec.utils
import numpy as np
from pandas.io.common import is_fsspec_url
from pydantic import BaseModel
from pydantic import ByteSize
from pydantic import Extra
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic.color import Color
from shapely.geometry import Polygon

import tiffslide
from pado.image.utils import IntPoint
from pado.image.utils import IntSize
from pado.image.utils import MPP
from pado.types import UrlpathLike
from pado.utils import cached_property
from tiffslide import TiffSlide
from tiffslide import __version__ as _tiffslide_version

if TYPE_CHECKING:
    import PIL


# --- metadata and info models ---

class ImageMetadata(BaseModel):
    """the common image metadata"""
    # essentials
    width: int
    height: int
    objective_power: str  # todo
    mpp_x: PositiveFloat
    mpp_y: PositiveFloat
    downsamples: List[PositiveFloat]
    vendor: Optional[str] = None
    # optionals
    comment: Optional[str] = None
    quickhash1: Optional[str] = None
    background_color: Optional[Color] = None
    bounds_x: Optional[PositiveInt] = None
    bounds_y: Optional[PositiveInt] = None
    bounds_width: Optional[PositiveInt] = None
    bounds_height: Optional[PositiveInt] = None
    # extra
    extra_json: Optional[str] = None


class FileInfo(BaseModel):
    """information related to the file on disk"""
    size_bytes: ByteSize
    md5_computed: Optional[str] = None
    time_last_access: Optional[datetime] = None
    time_last_modified: Optional[datetime] = None
    time_status_changed: Optional[datetime] = None


class PadoInfo(BaseModel):
    """information regarding the file loading"""
    urlpath: str
    pado_image_backend: str
    pado_image_backend_version: str


class SerializedImage(ImageMetadata, FileInfo, PadoInfo):
    class Config:
        extra = Extra.forbid


def urlpath_to_string(urlpath: UrlpathLike) -> str:
    raise NotImplementedError("todo")


class Image:
    """pado.img.Image is a wrapper around whole slide image data"""

    def __init__(
        self,
        urlpath: UrlpathLike,
        *,
        load_metadata: bool = False,
        load_file_info: bool = False,
        checksum: bool = False,
    ):
        """instantiate an image from an urlpath"""
        self.urlpath = urlpath
        self._metadata: Optional[ImageMetadata] = None
        self._file_info: Optional[FileInfo] = None

        # file handles
        self._ctx: Optional[ExitStack] = None
        self._slide: Optional[TiffSlide] = None

        # optional load on init
        if load_metadata or load_file_info or checksum:
            with self:
                if load_metadata:
                    self.load_metadata()
                if load_file_info or checksum:
                    self.load_file_info(checksum=checksum)

    @classmethod
    def from_obj(cls, obj) -> Image:
        """instantiate an image from an object, i.e. a pd.Series"""
        md = SerializedImage.parse_obj(obj)
        # get metadata
        metadata = ImageMetadata.parse_obj(md)
        file_info = FileInfo.parse_obj(md)
        pado_info = PadoInfo.parse_obj(md)
        # get extra data
        inst = cls(pado_info.urlpath)
        inst._metadata = metadata
        inst._file_info = file_info
        # todo: warn if tiffslide version difference
        # pado_info ...
        return inst

    def to_record(self) -> SerializedImage:
        pado_info = PadoInfo(
            urlpath=urlpath_to_string(self.urlpath),
            pado_image_backend=TiffSlide.__class__.__qualname__,
            pado_image_backend_version=_tiffslide_version,
        )
        return SerializedImage.parse_obj({
            **pado_info.dict(),
            **self.metadata.dict(),
            **self.file_info.dict(),
        })

    def __enter__(self) -> Image:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def open(self) -> Image:
        """open an image instance"""
        if not self._ctx:
            self._ctx = ctx = ExitStack()
            if is_fsspec_url(self.urlpath):
                urlpath_or_file = ctx.enter_context(fsspec.open(self.urlpath))
            else:
                urlpath_or_file = self.urlpath
            # noinspection PyTypeChecker
            self._slide = ctx.enter_context(TiffSlide(urlpath_or_file))
        return self

    def close(self):
        """close and image instance"""
        if self._ctx:
            self._ctx.close()
            self._slide = None
            self._ctx = None

    def __repr__(self):
        return f"{type(self).__name__}({self.urlpath!r})"

    def load_metadata(self, *, force: bool = False) -> None:
        """load the metadata from the file"""
        if self._metadata is None or force:
            if self._slide is None:
                raise RuntimeError("must be called on opened image")

            slide = self._slide
            props = slide.properties
            dimensions = slide.dimensions

            _used_keys = {}
            def pget(key): return _used_keys.setdefault(key, props.get(key))

            self._metadata = ImageMetadata(
                width=dimensions[0],
                height=dimensions[1],
                objective_power=pget(tiffslide.PROPERTY_NAME_OBJECTIVE_POWER),
                mpp_x=pget(tiffslide.PROPERTY_NAME_MPP_X),
                mpp_y=pget(tiffslide.PROPERTY_NAME_MPP_Y),
                downsamples=list(slide.level_downsamples),
                vendor=pget(tiffslide.PROPERTY_NAME_VENDOR),
                background_color=pget(tiffslide.PROPERTY_NAME_BACKGROUND_COLOR),
                quickhash1=pget(tiffslide.PROPERTY_NAME_QUICKHASH1),
                comment=pget(tiffslide.PROPERTY_NAME_COMMENT),
                bounds_x=pget(tiffslide.PROPERTY_NAME_BOUNDS_X),
                bounds_y=pget(tiffslide.PROPERTY_NAME_BOUNDS_Y),
                bounds_width=pget(tiffslide.PROPERTY_NAME_BOUNDS_WIDTH),
                bounds_height=pget(tiffslide.PROPERTY_NAME_BOUNDS_HEIGHT),
                extra_json=json.dumps({
                    key: value for key, value in sorted(props.items())
                    if key not in _used_keys
                })
            )

    def load_file_info(self, *, force: bool = False, checksum: bool = False) -> None:
        """load the file information from the file"""
        if self._file_info is None or force:
            if self._slide is None:
                raise RuntimeError("must be called on opened image")

            if isinstance(self.urlpath, str):
                fs, _, path = fsspec.get_fs_token_paths(self.urlpath)
            elif isinstance(self.urlpath, fsspec.core.OpenFile):
                fs = self.urlpath.fs
                path = self.urlpath.path
            else:
                raise NotImplementedError("todo")

            if checksum:
                _checksum = fs.checksum(path)
            else:
                _checksum = None

            info = fs.info(path)
            self._file_info = FileInfo(
                size_bytes=info['size'],
                md5_computed=_checksum,
                time_last_access=info.get('atime'),
                time_last_modified=info.get('mtime'),
                time_status_changed=info.get('created'),
            )

    @property
    def metadata(self) -> ImageMetadata:
        """the image metadata"""
        if self._metadata is None:
            # we need to load the image metadata
            if self._slide is None:
                raise RuntimeError(f"{self} not opened and not in context manager")
            self.load_metadata()
        return self._metadata

    @property
    def file_info(self) -> FileInfo:
        """stats regarding the image file"""
        if self._file_info is None:
            # we need to load the file_info data
            if self._slide is None:
                raise RuntimeError(f"{self} not opened and not in context manager")
            self.load_file_info(checksum=False)
        return self._file_info

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> Dict[int, IntSize]:
        dims = self._slide.level_dimensions
        down = self._slide.level_downsamples
        mpp0 = self.mpp
        return {
            lvl: IntSize(x, y, mpp0.scale(ds))
            for lvl, ((x, y), ds) in enumerate(zip(dims, down))
        }

    @property
    def level_mpp(self) -> Dict[int, MPP]:
        return {
            lvl: self.mpp.scale(ds)
            for lvl, ds in enumerate(self._slide.level_downsamples)
        }

    @property
    def mpp(self) -> MPP:
        return MPP(self.metadata.mpp_x, self.metadata.mpp_y)

    @property
    def dimensions(self) -> IntSize:
        return IntSize(
            x=self.metadata.width,
            y=self.metadata.height,
            mpp=self.mpp,
        )

    def get_thumbnail(self, size: IntSize) -> PIL.Image.Image:
        return self._slide.get_thumbnail(size=(size.width, size.height))

    def get_array(
        self,
        location: IntPoint,
        region: IntSize,
    ) -> np.ndarray:
        """return array"""
        if not isinstance(location, IntPoint) or location.mpp is None:
            raise ValueError(f"must provide location as IntPoint with mpp, got: {location!r}")
        if not isinstance(region, IntSize) or region.mpp is None:
            raise ValueError(f"must provide region as IntSize with mpp, got: {location!r}")

        '''
        if mpp_xy is None and level is not None:
            if level == 0:
                img = self._image_backend.get_region(location_xy, region_wh, level=level)
            else:
                lvl_mpp = self._image_backend.level_mpp_map[level]
                lvl0_mpp = self._image_backend.level0_mpp
                lvl0_xy = tuple_round(scale_xy(location_xy, current=lvl0_mpp, target=lvl_mpp))
                img = self._image_backend.get_region(lvl0_xy, region_wh, level=level)

        elif level is None and mpp_xy is not None:
            # mpp_xy is set
            assert self.metadata[S_MPP_X] == self.metadata[S_MPP_Y]
            lvl0_mpp = self._image_backend.level0_mpp
            lvl0_xy = tuple_round(
                scale_xy(location_xy, current=mpp_xy, target=lvl0_mpp)
            )

            mpp_map = self._image_backend.level_mpp_map
            for lvl_best, mpp_best in mpp_map.items():
                if mpp_xy[0] >= mpp_best[0]:
                    break
            else:
                raise NotImplementedError(f"requesting a smaller mpp than provided in the image {mpp_xy!r}")

            if mpp_xy == mpp_best:
                img = self._image_backend.get_region(lvl0_xy, region_wh, level=lvl_best)

            else:
                assert mpp_best[0] < mpp_xy[0]
                region_wh_best = tuple_round(
                    scale_xy(region_wh, current=mpp_xy[0], target=mpp_best[0])
                )
                assert region_wh_best[0] > region_wh[0]
                img = self._image_backend.get_region(lvl0_xy, region_wh_best, level=lvl_best, downsize_to=region_wh)

        else:
            raise ValueError("cannot specify both level and mpp_xy")

        return np.array(img)
        '''
        raise NotImplementedError("todo")


class TileIterator:
    def __init__(self, image, tile_size, mpp_xy):
        self._image: Image = image

    def __iter__(self):
        return self

    def __next__(self):
        """
        resample_factor_lvl_0 = self.target_mpp / slide_mpp

        if resample_factor_lvl_0 < 1.0:
            # fixme: ??? up-sample ???
            raise ValueError("lvl0 resolution is lower than target_mpp requires")

        lvl_b = self.slide.get_best_level_for_downsample(resample_factor_lvl_0)

        resample_factor = resample_factor_lvl_0
        if lvl_b > 0:
            # resample according to image lvl we're using for extraction
            resample_factor /= self.slide.level_downsamples[lvl_b]

        tile = np.array(self.tile_size, dtype=int)
        tile_0 = (tile * resample_factor_lvl_0).astype(int)  # floor
        tile_b = (tile * resample_factor).astype(int)
        s_dim = self.slide.dimensions

        # read tiles from slide
        r0, r1 = range(0, s_dim[0], tile_0[0]), range(0, s_dim[1], tile_0[1])
        idx_iter = itertools.product(enumerate(r1), enumerate(r0))

        for (idx_y, y), (idx_x, x) in tqdm(
                idx_iter,
                total=len(r0) * len(r1),
                desc=f"tiles of {self._image_path.name}:",
                leave=False,
        ):
            tile_image = self.slide.read_region((x, y), lvl_b, tuple(tile_b))
            tile_image = tile_image.convert("RGB")
            tile_image = tile_image.resize(self.tile_size, PIL.Image.BILINEAR)

            yield {
                "image": tile_image,
                "metadata": {
                    "size": self.tile_size,
                    "name": self._image_path.stem,
                    "idx_x": idx_x,
                    "idx_y": idx_y,
                    "slide_x": x,
                    "slide_y": y,
                },
            }
        """
        ...


class Tile:
    """pado.img.Tile abstracts rectangular regions in whole slide image data"""
    def __init__(
        self,
        mpp_xy: Tuple[float, float],
        bounds: Tuple[int, int, int, int],
        data_array: Optional[np.ndarray] = None,
        data_bytes: Optional[bytes] = None,
        parent: Optional[Image] = None,
        mask: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        tissue: Optional[np.ndarray] = None,
    ):
        self.mpp_xy = mpp_xy
        self.bounds = bounds
        self.data_array = data_array
        self.data_bytes = data_bytes
        self.parent = parent
        self.mask = mask
        self.labels = labels
        self.tissue = tissue

    @cached_property
    def shape(self) -> Polygon:
        return Polygon.from_bounds(*self.bounds)

    @property
    def size(self):
        return self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1]

    def bounds_at_mpp(self, mpp_xy: Tuple[float, float], *, as_int: bool = True):
        """return the tile bounds at another mpp level"""
        sx, sy = self.mpp_xy
        ox, oy = mpp_xy
        rx, ry = ox / sx, oy / sy
        x0, y0, x1, y1 = self.bounds
        if not as_int:
            return x0 * rx, y0 * ry, x1 * rx, y1 * ry
        else:
            return int(x0 * rx), int(y0 * ry), int(x1 * rx), int(y1 * ry)

    def shape_at_mpp(self, mpp_xy: Tuple[float, float], *, as_int: bool = True):
        """return the tile shape at another mpp level"""
        return Polygon.from_bounds(*self.bounds_at_mpp(mpp_xy, as_int=as_int))

    def size_at_mpp(self, mpp_xy: Tuple[float, float], *, as_int: bool = True):
        """return the tile size at another mpp level"""
        x0, y0, x1, y1 = self.bounds_at_mpp(mpp_xy, as_int=as_int)
        return x1 - x0, y1 - y0

    @property
    def x0y0(self):
        return self.bounds[:2]
    tl = x0y0

    @property
    def wh(self):
        return self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1]

