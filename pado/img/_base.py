"""pado image abstraction to hide image loading implementation"""
import os
from abc import ABC
from abc import abstractmethod
from contextlib import ExitStack
from contextlib import suppress
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type

import fsspec
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from pado.img.utils import scale_xy
from pado.img.utils import tuple_round
from pado.utils import cached_property


# fmt: off
# essential
S_WIDTH               = 'width'
S_HEIGHT              = 'height'
S_OBJECTIVE_POWER     = 'objective_power'
S_MPP_X               = 'mpp_x'
S_MPP_Y               = 'mpp_y'
S_DOWNSAMPLES         = 'downsamples'
S_VENDOR              = 'vendor'
# optional
S_COMMENT             = 'slide_comment'
S_QUICKHASH1          = 'quickhash1'
S_BACKGROUND_COLOR    = 'background_color'
S_BOUNDS_X            = 'bounds_x'
S_BOUNDS_Y            = 'bounds_y'
S_BOUNDS_WIDTH        = 'bounds_width'
S_BOUNDS_HEIGHT       = 'bounds_height'
# file_info
F_SIZE_BYTES          = 'size_bytes'
F_MD5_COMPUTED        = 'md5_computed'
# F_TIME_LAST_ACCESS  = 'atime'  # not provided in every fsspec implementation
F_TIME_LAST_MODIFIED  = 'mtime'
F_TIME_STATUS_CHANGED = 'ctime'
# fmt: on

# conveniently group the above defined slide constants into a namespace
N = SimpleNamespace(**{c: k for c, k in globals().items() if c.startswith("S_")})


class UnsupportedImageFormat(Exception):
    pass


class ImageBackend(ABC):
    """the backend image class used to read images"""
    # this is just a way to ultimately remove openslide from the equation
    # and also exactly the reason, why this is mapping almost exactly to
    # openslide for now.

    def __init__(self, fspath):
        self._fspath = fspath
        self._fs: fsspec.AbstractFileSystem = fsspec.core.get_fs_token_paths(fspath)[0]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def image_metadata(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def file_stats(self, *, checksum: bool = True):
        _checksum = None
        if checksum:
            _checksum = self._fs.checksum(self._fspath)

        stat = self._fs.info(self._fspath)
        return {
            F_SIZE_BYTES: stat.get('size'),
            # F_TIME_LAST_ACCESS: stat.st_atime,
            F_TIME_LAST_MODIFIED: stat.get('mtime'),
            F_TIME_STATUS_CHANGED: stat.get('created'),
            F_MD5_COMPUTED: _checksum
        }

    @property
    @abstractmethod
    def level_mpp_map(self) -> Dict[int, Tuple[float, float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def level0_mpp(self) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_size(self, level: int = 0) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def get_region(
        self,
        location_xy: Tuple[int, int],
        region_wh: Tuple[int, int],
        level: int = 0,
        *,
        downsize_to: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        raise NotImplementedError


# === image backend loading ===========================================

def _openslide():
    from pado.img._impl_openslide import OpenSlideImageBackend
    return OpenSlideImageBackend

def _imageslide():
    from pado.img._impl_openslide import ImageSlideImageBackend
    return ImageSlideImageBackend

def _tiffslide():
    from pado.img._impl_tifffile import TiffFileImageBackend
    return TiffFileImageBackend

def _initialize_image_backend_preference(backends = None):
    global _IMAGE_BACKENDS
    if backends is None:
        return list(_IMAGE_BACKENDS)
    if isinstance(backends, str):
        backends = backends.split(",")
    backends = [b.lower() for b in backends]
    assert backends
    assert set(backends).issubset(_IMAGE_BACKENDS)
    return backends

_IMAGE_BACKENDS = {
    'openslide': _openslide,
    'tiffslide': _tiffslide,
    'imageslide': _imageslide,
}
_IMAGE_BACKEND_ORDER = _initialize_image_backend_preference(
    os.environ.get("PADO_IMAGE_BACKEND")
)


def set_image_backend_preference(backends):
    """set one or more preferred image backends"""
    global _IMAGE_BACKEND_ORDER
    bs = _initialize_image_backend_preference(backends)
    _IMAGE_BACKEND_ORDER[:] = bs


def get_image_backend() -> Iterable[Type[ImageBackend]]:
    """iterate the ImageBackends in order"""
    global _IMAGE_BACKEND_ORDER
    global _IMAGE_BACKENDS
    for backend in _IMAGE_BACKEND_ORDER:
        with suppress(ImportError):
            yield _IMAGE_BACKENDS[backend]()


class Image:
    """pado.img.Image is a wrapper around whole slide image data"""
    _fields = (  # minimum required fields
        'size_bytes',
        'mtime',
        'ctime',
        'md5_computed',
        'width',
        'height',
        'objective_power',
        'mpp_x',
        'mpp_y',
        'downsamples',
        'vendor',
        'background_color',
        'quickhash1',
        'slide_comment',
        'bounds_x',
        'bounds_y',
        'bounds_width',
        'bounds_height',
        'pado_image_backend',
    )

    def __init__(self, fspath, *, metadata: Optional[Dict[str, Any]] = None):
        self.fspath = os.fspath(fspath)
        self._metadata = metadata

        # file handling
        self._image_backend: Optional[ImageBackend] = None
        self._image_cm: Optional[ExitStack] = None

    def __repr__(self):
        return f"{type(self).__name__}({self.fspath!r})"

    @classmethod
    def from_dict(cls, dct) -> 'Image':
        if isinstance(dct, dict):
            pass
        elif isinstance(dct, pd.Series):
            dct = dct.to_dict()
        elif isinstance(dct, tuple) and hasattr(dct, '_fields'):
            # noinspection PyProtectedMember
            dct = dict(zip(dct._fields, dct))
        else:
            dct = dict(dct)
        path = dct.pop('fspath')
        return cls(path, metadata=dct)

    def to_dict(self) -> dict:
        if self._metadata is None:
            with self:
                self._read_metadata_from_image()
        dct = self.metadata.copy()
        dct['fspath'] = self.fspath
        return dct

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        if not self._image_backend:
            self._image_cm = ExitStack()
            for image_backend in get_image_backend():
                inst = image_backend(self.fspath)
                try:
                    # noinspection PyTypeChecker
                    inst = self._image_cm.enter_context(inst)
                except (UnsupportedImageFormat, RuntimeError):
                    continue
                else:
                    break
            else:
                raise RuntimeError(f"no suitable backend for {self}")
            self._image_backend = inst
        return self

    def close(self):
        if self._image_cm:
            self._image_cm.close()
            self._image_cm = None
            self._image_backend = None

    def _read_metadata_from_image(self, checksum=False):
        assert self._image_backend is not None
        # get the metadata
        md = self._image_backend.image_metadata()
        # get fs_info data first
        fs = self._image_backend.file_stats(checksum=checksum)

        assert set(md).isdisjoint(fs), "no overlap between fs and md data"

        ib = {'pado_image_backend': self._image_backend.__class__.__qualname__}

        # combine them
        self._metadata = dict(**fs, **md, **ib)
        return self._metadata

    @property
    def metadata(self):
        if self._metadata is not None:
            return self._metadata

        # we need to load the image metadata
        if self._image_backend is None:
            if self._image_cm is None:
                raise RuntimeError(f"{self} not opened and not in context manager")
            self.open()

        return self._read_metadata_from_image()

    def __iter__(self):
        level0_tilesize = 512  # todo: ideally level0 size
        return TileIterator(self, tile_size=level0_tilesize, mpp_xy=self.mpp)

    @property
    def mpp(self):
        return self._image_backend.level0_mpp

    @property
    def levels(self):
        return tuple(self._image_backend.level_mpp_map)

    def get_size(self, mpp_xy: Optional[Tuple[float, float]] = None, level: Optional[int] = None) -> Tuple[int, int]:
        if mpp_xy is not None and level is not None:
            raise ValueError("can only specify one of: 'mpp_xy' and 'level'")
        elif mpp_xy is None:
            level = level or 0  # set 0 if none
            return self._image_backend.get_size(level)
        else:
            size_xy = self._image_backend.get_size(0)
            lvl0_mpp = float(self.metadata[S_MPP_X]), float(self.metadata[S_MPP_Y])
            return tuple_round(scale_xy(size_xy, current=lvl0_mpp, target=mpp_xy))

    def get_region(self, location_xy: Tuple[int, int], region_wh: Tuple[int, int], *,
                   mpp_xy: Optional[Tuple[float, float]] = None, level: Optional[int] = None) -> np.array:
        # location_xy is not in level 0 coordinates
        if mpp_xy is level is None:
            level = 0

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

