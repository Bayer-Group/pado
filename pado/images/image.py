"""pado image abstraction to hide image loading implementation"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tiffslide
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic import ByteSize
from pydantic import Extra
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import validator
from pydantic.color import Color
from tifffile import ZarrTiffStore
from tiffslide import TiffSlide

# noinspection PyProtectedMember
from tiffslide._zarr import get_zarr_chunk_sizes

from pado.images.utils import MPP
from pado.images.utils import IntPoint
from pado.images.utils import IntSize
from pado.io.checksum import Checksum
from pado.io.checksum import compare_checksums
from pado.io.checksum import compute_checksum
from pado.io.files import update_fs_storage_options
from pado.io.files import urlpathlike_get_fs_cls
from pado.io.files import urlpathlike_is_localfile
from pado.io.files import urlpathlike_local_via_fs
from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_fsspec
from pado.io.files import urlpathlike_to_string
from pado.io.paths import get_dataset_fs
from pado.types import UrlpathLike

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image

    from pado.dataset import PadoDataset

try:
    import cv2
except ImportError:
    cv2 = None


_log = logging.getLogger(__name__)


# --- metadata and info models ---


class ImageMetadata(BaseModel):
    """the common image metadata"""

    # essentials
    width: int
    height: int
    objective_power: Optional[str]  # todo
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

    @validator("downsamples", pre=True)
    def downsamples_as_list(cls, v):
        # this is stored as array in parquet
        return list(v)


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


class _SerializedImage(ImageMetadata, FileInfo, PadoInfo):
    class Config:
        extra = Extra.forbid


class Image:
    """pado.img.Image is a wrapper around whole slide image data"""

    __slots__ = (
        "urlpath",
        "_metadata",
        "_file_info",
        "_slide",
    )  # prevent attribute errors during refactor
    __fields__ = _SerializedImage.__fields__

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
        self._slide: Optional[TiffSlide] = None

        # optional load on init
        if load_metadata or load_file_info or checksum:
            with self:
                if load_metadata:
                    self._metadata = self._load_metadata()
                if load_file_info or checksum:
                    self._file_info = self._load_file_info(checksum=checksum)

    @classmethod
    def from_obj(cls, obj) -> Image:
        """instantiate an image from an object, i.e. a pd.Series"""
        md = _SerializedImage.parse_obj(obj)
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

    def to_record(self, *, urlpath_ignore_options: Collection[str] = ()) -> dict:
        """return a record for serializing"""
        pado_info = PadoInfo(
            urlpath=urlpathlike_to_string(
                self.urlpath, ignore_options=urlpath_ignore_options
            ),
            pado_image_backend=TiffSlide.__class__.__qualname__,
            pado_image_backend_version=tiffslide.__version__,
        )
        return _SerializedImage.parse_obj(
            {
                **pado_info.dict(),
                **self.metadata.dict(),
                **self.file_info.dict(),
            }
        ).dict()

    def __enter__(self) -> Image:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(
        self,
        *,
        via: AbstractFileSystem | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> Image:
        """open an image instance

        This will instantiate the filesystem. Dependent on the
        filesystem this will establish connections to servers, etc.
        If open has been called, following calls will be no-ops.

        Parameters
        ----------
        via:
            allows to provide a filesystem that will be used instead of
            the Image.urlpath's filesystem to access the path.
        storage_options:
            allows providing storage options for the filesystem used to
            access the image.

        Returns
        -------
        self:
            returns the opened image instance
        """
        if not self._slide:
            if via is None or isinstance(via, MemoryFileSystem):
                of = urlpathlike_to_fsspec(
                    self.urlpath, storage_options=storage_options
                )
            elif isinstance(via, AbstractFileSystem):
                of = urlpathlike_local_via_fs(
                    self.urlpath,
                    fs=update_fs_storage_options(via, storage_options=storage_options),
                )
            else:
                raise TypeError(
                    f"via not an AbstractFileSystem, got {type(via).__name__}"
                )
            try:
                self._slide = TiffSlide(of)
            except Exception as e:
                _log.error(f"{self.urlpath!r} with error {e!r}")
                self.close()
                raise
        return self

    def via(
        self,
        ds: PadoDataset,
        *,
        storage_options: dict[str, Any] | None = None,
    ) -> Image:
        """open an image instance via a pado dataset

        Similar behavior to .open() with the difference that only if
        the dataset is accessed remotely and the images are referenced
        locally (so on the same remote) will the image be accessed via
        the dataset filesystem.

        A common example is a pado dataset stored on a server with the
        images stored next to it on the server filesystem. If this
        dataset is now accessed via ssh, the images will be accessible
        via ssh too.

        Parameters
        ----------
        ds:
            this pado dataset's filesystem will be used for access
        storage_options:
            allows providing storage options for the filesystem used to
            access the image.

        Returns
        -------
        self:
            returns the opened image instance
        """
        ds_fs = get_dataset_fs(ds)
        # check if we are accessing a dataset remotely, that has references to
        # files locally. For now access via ssh is the primary use case for this.
        if not isinstance(ds_fs, LocalFileSystem):
            im_fs_cls = urlpathlike_get_fs_cls(self.urlpath)
            if issubclass(im_fs_cls, LocalFileSystem):
                self.open(via=ds_fs, storage_options=storage_options)
                return self

        self.open()  # to make .via()'s behavior consistent we have to call open here
        return self

    def close(self):
        """close and image instance"""
        if self._slide:
            self._slide.close()
            self._slide = None

    def __repr__(self):
        return f"{type(self).__name__}({self.urlpath!r})"

    def __eq__(self, other: Any) -> bool:
        """compare if two images are identical"""
        if not isinstance(other, Image):
            return False
        # if checksum available for both
        if self.file_info.md5_computed and other.file_info.md5_computed:
            try:
                return compare_checksums(
                    self.file_info.md5_computed, other.file_info.md5_computed
                )
            except ValueError:
                pass
        if self.file_info.size_bytes != other.file_info.size_bytes:
            return False
        return self.metadata == other.metadata

    def _load_metadata(self, *, force: bool = False) -> ImageMetadata:
        """load the metadata from the file"""
        if self._metadata is None or force:
            if self._slide is None:
                raise RuntimeError(f"{self!r} not opened and not in context manager")

            slide = self._slide
            props = slide.properties
            dimensions = slide.dimensions

            _used_keys: Dict[str, Any] = {}

            def pget(key):
                return _used_keys.setdefault(key, props.get(key))

            return ImageMetadata(
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
                extra_json=json.dumps(
                    {
                        key: value
                        for key, value in sorted(props.items())
                        if key not in _used_keys
                    }
                ),
            )
        else:
            return self._metadata

    def _load_file_info(
        self, *, force: bool = False, checksum: bool = False
    ) -> FileInfo:
        """load the file information from the file"""
        if self._file_info is None or force:
            if self._slide is None:
                raise RuntimeError(f"{self!r} not opened and not in context manager")

            fs, path = urlpathlike_to_fs_and_path(self.urlpath)
            if checksum:
                checksums = compute_checksum(self.urlpath, available_only=not force)
                _checksum = Checksum.join_checksums(checksums)
            else:
                _checksum = None

            info = fs.info(path)
            return FileInfo(
                size_bytes=info["size"],
                md5_computed=_checksum,
                time_last_access=info.get("atime"),
                time_last_modified=info.get("mtime"),
                time_status_changed=info.get("created"),
            )
        else:
            return self._file_info

    @property
    def metadata(self) -> ImageMetadata:
        """the image metadata"""
        if self._metadata is None:
            # we need to load the image metadata
            if self._slide is None:
                raise RuntimeError(f"{self!r} not opened and not in context manager")
            self._metadata = self._load_metadata()
        return self._metadata

    @property
    def file_info(self) -> FileInfo:
        """stats regarding the image file"""
        if self._file_info is None:
            # we need to load the file_info data
            if self._slide is None:
                raise RuntimeError(f"{self!r} not opened and not in context manager")
            self._file_info = self._load_file_info(checksum=False)
        return self._file_info

    @property
    def level_count(self) -> int:
        if self._slide is None:
            raise RuntimeError(f"{self!r} not opened and not in context manager")
        return self._slide.level_count

    @property
    def level_dimensions(self) -> Dict[int, IntSize]:
        if self._slide is None:
            raise RuntimeError(f"{self!r} not opened and not in context manager")
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
            lvl: self.mpp.scale(ds) for lvl, ds in enumerate(self.metadata.downsamples)
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

    def get_thumbnail(self, size: Union[IntSize, Tuple[int, int]]) -> PIL.Image.Image:
        if self._slide is None:
            raise RuntimeError(f"{self!r} not opened and not in context manager")
        if isinstance(size, tuple):
            _, _ = size
        elif isinstance(size, IntSize):
            size = size.as_tuple()
        else:
            raise TypeError(
                f"expected tuple or IntSize, got {size!r} of cls {type(size).__name__}"
            )
        return self._slide.get_thumbnail(size=size, use_embedded=True)

    def get_array(
        self,
        location: IntPoint,
        region: IntSize,
        level: int,
        *,
        runtime_type_checks: bool = True,
    ) -> np.ndarray:
        """return array from a defined level"""
        if runtime_type_checks:
            if self._slide is None:
                raise RuntimeError(f"{self!r} not opened and not in context manager")

            # location
            if not isinstance(location, IntPoint):
                raise TypeError(
                    f"location requires IntPoint, got: {location!r} of {type(location).__name__}"
                )
            elif location.mpp is not None and location.mpp != self.mpp:
                _guess = next(  # improve error for user
                    (idx for idx, mpp in self.level_mpp.items() if mpp == location.mpp),
                    "level-not-in-image",
                )
                raise ValueError(
                    f"location not at level 0, got {location!r} at {_guess}"
                )

            # level (indirectly)
            try:
                level_mpp = self.level_mpp[level]
            except KeyError:
                raise ValueError(f"level error: 0 <= {level} <= {self.level_count}")

            # region
            if not isinstance(region, IntSize):
                raise TypeError(
                    f"region requires IntSize, got: {region!r} of {type(region).__name__}"
                )
            elif region.mpp is not None and region.mpp != level_mpp:
                _guess = next(  # improve error for user
                    (idx for idx, mpp in self.level_mpp.items() if mpp == region.mpp),
                    "level-not-in-image",
                )
                raise ValueError(
                    f"region not at level {level}, got {region!r} at {_guess}"
                )

        if self._slide is None:
            raise RuntimeError(f"{self!r} not opened and not in context manager")

        return self._slide.read_region(
            location.as_tuple(), level, region.as_tuple(), as_array=True
        )

    def get_array_at_mpp(
        self, location: IntPoint, region: IntSize, target_mpp: MPP
    ) -> np.ndarray:
        """return array from a defined mpp and a position (in the target mpp)"""

        if location.mpp != target_mpp:
            raise ValueError(
                f"location.mpp != target_mpp -> {location.mpp!r} != {target_mpp!r}"
            )
        if target_mpp.x != target_mpp.y:
            raise NotImplementedError("currently assuming same x and y mpp")

        if region.mpp is None:
            pass
        elif region.mpp != target_mpp:
            raise ValueError(
                f"region.mpp != target_mpp -> {region.mpp!r} != {target_mpp!r}"
            )

        # we find the corresponding location at level0
        lvl0_xy = _scale_xy(
            location,
            mpp_current=target_mpp,
            mpp_target=self.mpp,
        )
        region_wh = region.as_tuple()

        for lvl_best, mpp_best in self.level_mpp.items():
            if target_mpp > mpp_best or target_mpp == mpp_best:
                break
        else:
            raise NotImplementedError(
                f"requesting a smaller mpp {target_mpp!r} "
                f"than provided in the image {self.level_mpp.items()!r}"
            )

        if target_mpp == mpp_best:
            # no need to rescale
            array = self._slide.read_region(
                location=lvl0_xy, level=lvl_best, size=region_wh, as_array=True
            )
        else:
            # we need to rescale to the target_mpp
            region_best = _scale_xy(
                region, mpp_current=location.mpp, mpp_target=mpp_best
            )

            array = self._slide.read_region(
                location=lvl0_xy, level=lvl_best, size=region_best, as_array=True
            )

            if array.shape[0:2:-1] != region_wh:
                array = cv2.resize(array, dsize=region_wh)

        return array

    def get_zarr_store(
        self,
        level: int,
        *,
        chunkmode: int = 0,
        zattrs: dict[str, Any] | None = None,
    ) -> ZarrTiffStore:
        """return the entire level as a zarr store"""
        if self._slide is None:
            raise RuntimeError(f"{self!r} not opened and not in context manager")
        return self._slide.ts_tifffile.aszarr(
            key=None,
            series=None,
            level=level,
            chunkmode=chunkmode,
            zattrs=zattrs,
        )

    def get_chunk_sizes(
        self,
        level: int = 0,
    ) -> NDArray[np.int]:
        """return a chunk bytesize array"""
        if self._slide is None:
            raise RuntimeError(f"{self!r} not opened and not in context manager")
        axes = self._slide.properties["tiffslide.series-axes"]
        if axes == "YXS":
            sum_axis = 2
        elif axes == "CYX":
            sum_axis = 0
        else:
            raise NotImplementedError(f"axes: {axes!r}")
        return get_zarr_chunk_sizes(
            self._slide.zarr_group, level=level, sum_axis=sum_axis
        )

    def is_local(self, must_exist=True) -> bool:
        """Return True if the image is stored locally"""
        return urlpathlike_is_localfile(self.urlpath, must_exist=must_exist)


def _scale_xy(
    to_transform: Union[IntPoint, IntSize], mpp_current: MPP, mpp_target: MPP
):
    pos_x, pos_y = to_transform.as_tuple()
    mpp_x_current, mpp_y_current = mpp_current.as_tuple()
    mpp_x_target, mpp_y_target = mpp_target.as_tuple()
    x = int(round(pos_x * mpp_x_current / mpp_x_target))
    y = int(round(pos_y * mpp_y_current / mpp_y_target))
    return x, y
