"""pado image abstraction to hide image loading implementation"""
import hashlib
import importlib
from abc import ABC
from abc import abstractmethod
from contextlib import ExitStack
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
from shapely.geometry import Polygon

from pado.fileutils import hash_file
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
F_TIME_LAST_ACCESS    = 'atime'
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

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    @abstractmethod
    def path(self):
        raise

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
        md5_computed = None
        if checksum:
            md5_computed = hash_file(self._path, hasher=hashlib.md5)

        stat = self._path.stat()
        return {
            F_SIZE_BYTES: stat.st_size,
            F_TIME_LAST_ACCESS: stat.st_atime,
            F_TIME_LAST_MODIFIED: stat.st_mtime,
            F_TIME_STATUS_CHANGED: stat.st_ctime,
            F_MD5_COMPUTED: md5_computed
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
    ) -> np.array:
        raise NotImplementedError


def get_image_backend():
    """iterate the ImageBackends in order"""
    with suppress(ImportError):
        from pado.img._impl_openslide import OpenSlideImageBackend
        yield OpenSlideImageBackend


class Image:
    """pado.img.Image is a wrapper around whole slide image data"""

    def __init__(self, path, *, metadata: Optional[Dict[str, Any]] = None):
        self._path = Path(path)
        self._metadata = metadata

        # file handling
        self._image_backend: Optional[ImageBackend] = None
        self._image_cm: Optional[ExitStack] = None

    def __enter__(self):
        self._image_cm = ExitStack()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._image_cm:
            self._image_cm.close()
            self._image_cm = None
        return False

    def open(self):
        if not self._image_backend:
            for image_backend in get_image_backend():
                inst = image_backend(self._path)
                try:
                    inst.open()
                except (UnsupportedImageFormat, RuntimeError):
                    continue
                else:
                    break
            else:
                raise RuntimeError(f"no suitable backend for {self}")
            self._image_backend = inst
        return self

    def close(self):
        if self._image_backend:
            self._image_backend.close()
            self._image_backend = None

    def _read_metadata_from_image(self, checksum=False):
        assert self._image_backend is not None
        # get the metadata
        md = self._image_backend.image_metadata()
        # get fs_info data first
        fs = self._image_backend.file_stats(checksum=checksum)

        assert set(md).isdisjoint(fs), "no overlap between fs and md data"

        ib = {'pado_image_backend': self._image_backend.__class__.__qualname__}
        print(self._path, ib['pado_image_backend'])

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
        return TileIterator(self)

    def get_size(self, mpp_xy: Optional[Tuple[float, float]] = None, level: Optional[int] = None) -> Tuple[int, int]:
        if mpp_xy is not None and level is not None:
            raise ValueError("can only specify one of: 'mpp_xy' and 'level'")
        elif mpp_xy is None:
            level = level or 0  # set 0 if none
            return self._slide.level_dimensions[level]
        else:
            return tuple_round(
                scale_xy(self._slide.dimensions, current=self.level0_mpp, target=mpp_xy)
            )

    def get_region(self, location_xy: Tuple[int, int], region_wh: Tuple[int, int],
                   mpp_xy: Optional[Tuple[float, float]] = None, level: Optional[int] = None) -> np.array:
        # location_xy is not in level 0 coordinates
        if mpp_xy is level is None:
            level = 0

        if mpp_xy is None and level is not None:
            if level == 0:
                img = self._slide.read_region(location_xy, level, region_wh)
            else:
                ds = self._slide.level_downsamples[level]
                lvl0_xy = scale_xy(location_xy, current=(ds, ds), target=(1, 1))
                img = self._slide.read_region(lvl0_xy, level, region_wh)

        elif level is None and mpp_xy is not None:
            # mpp_xy is set
            lvl0_mpp = self.level0_mpp
            lvl0_xy = tuple_round(
                scale_xy(location_xy, current=mpp_xy, target=lvl0_mpp)
            )
            ds_target = mpp_xy[0] / lvl0_mpp[0]
            assert ds_target == mpp_xy[1] / lvl0_mpp[1], "sanity check: downsamples need to be symmetric"
            lvl_best = self._slide.get_best_level_for_downsample(ds_target)
            ds_best = self._slide.level_downsamples[lvl_best]
            region_wh_best = tuple_round(
                scale_xy(region_wh, current=ds_target, target=ds_best)
            )
            img = self._slide.read_region(lvl0_xy, lvl_best, region_wh_best)
            img.thumbnail(region_wh)

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
    ):
        self.mpp_xy = mpp_xy
        self.bounds = bounds
        self.data_array = data_array
        self.data_bytes = data_bytes
        self.parent = parent
        self.mask = mask
        self.labels = labels

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

