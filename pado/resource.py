import contextlib
import hashlib
import platform
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen


class ImageResource(ABC):
    __slots__ = ("_image_id", "_resource")

    def __init__(self, image_id, resource, *mixin_args, **mixin_kwargs):
        super().__init__(*mixin_args, **mixin_kwargs)
        if isinstance(image_id, str):
            image_id = (image_id,)
        if not isinstance(image_id, (tuple, list)):
            raise TypeError(f"image_id not str or tuple, got {type(image_id)}")
        self._image_id = tuple(image_id)
        self._resource = resource

    @property
    def id(self) -> Tuple[str, ...]:
        return self._image_id

    @abstractmethod
    def open(self):
        """return a file like object"""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """return the size of the file"""
        ...


class MD5Resource:
    __slots__ = ("md5sum",)

    def __init__(self, *args, md5sum=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.md5sum = md5sum


class LocalImageResource(ImageResource, MD5Resource):
    __slots__ = ("_path",)

    def __init__(self, image_id, resource, **mixin_kwargs):
        if not isinstance(resource, (Path, str)):
            raise TypeError(f"resource not str or pathlib.Path, got {type(resource)}")
        super().__init__(image_id, resource, **mixin_kwargs)
        self._path = Path(resource)

    def open(self):
        return self._path.open("rb")

    @property
    def size(self):
        return self._path.stat().st_size


class RemoteImageResource(ImageResource, MD5Resource):
    __slots__ = ("_url", "_fp")

    def __init__(self, image_id, resource, **mixin_kwargs):
        super().__init__(image_id, resource, **mixin_kwargs)
        if not isinstance(resource, str):
            raise TypeError(f"url not str, got {type(resource)}")
        if urlparse(resource).scheme not in {"http", "https", "ftp"}:
            warnings.warn(f"untested scheme for url: '{resource}'")
        self._url = resource
        self._fp = None

    @contextlib.contextmanager
    def open(self):
        try:
            self._fp = urlopen(self._url)
            yield self._fp
        finally:
            self._fp.close()
            self._fp = None

    @property
    def size(self):
        try:
            return int(self._fp.info()["Content-length"])
        except (AttributeError, KeyError):
            return -1


_WINDOWS = platform.system() == "Windows"
_BLOCK_SIZE = {
    LocalImageResource.__name__: 1024 * 1024 if _WINDOWS else 1024 * 64,
    RemoteImageResource.__name__: 1024 * 8,
}


def copy_resource(
    resource: ImageResource, path: Path, progress_hook: Callable[[int, int], None]
):
    """copy an image resource to a local path"""
    md5hash = None
    # in case we provide an md5 build the hash incrementally
    if isinstance(resource, MD5Resource) and resource.md5sum:
        md5hash = hashlib.md5()

    with resource.open() as src, path.open("wb") as dst:
        src_size = resource.size
        bs = _BLOCK_SIZE[resource.__class__.__name__]
        src_read = src.read
        dst_write = dst.write
        read = 0

        if progress_hook:
            progress_hook(read, src_size)

        while True:
            buf = src_read(bs)
            if not buf:
                break
            read += len(buf)
            dst_write(buf)
            if md5hash:
                md5hash.update(buf)
            if progress_hook:
                progress_hook(read, src_size)

    if src_size >= 0 and read < src_size:
        raise RuntimeError(f"{resource.id}: could only copy {read} of {src_size} bytes")

    if md5hash and md5hash.hexdigest() != resource.md5sum:
        raise ValueError(f"{resource.id}: md5sum does not match provided md5")
