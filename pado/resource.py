import contextlib
import platform
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen


class ImageResource(ABC):
    def __init__(self, image_id, *_, **__):
        if isinstance(image_id, str):
            image_id = (image_id,)
        if not isinstance(image_id, (tuple, list)):
            raise TypeError(f"image_id not str or tuple, got {type(image_id)}")
        self._image_id = tuple(image_id)

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


class LocalImageResource(ImageResource):
    def __init__(self, image_id, path, *_, **__):
        super(LocalImageResource, self).__init__(image_id)
        if not isinstance(path, (Path, str)):
            raise TypeError(f"path not str or pathlib.Path, got {type(path)}")
        self._path = Path(path)

    def open(self):
        return self._path.open("rb")

    @property
    def size(self):
        return self._path.stat().st_size


class RemoteImageResource(ImageResource):
    def __init__(self, image_id, url, *_, **__):
        super(RemoteImageResource, self).__init__(image_id)
        if not isinstance(url, str):
            raise TypeError(f"url not str, got {type(url)}")
        if urlparse(url).scheme not in {"http", "https", "ftp"}:
            warnings.warn(f"untested scheme for url: '{url}'")
        self._url = url
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
            if progress_hook:
                progress_hook(read, src_size)

    if src_size >= 0 and read < src_size:
        raise RuntimeError(f"Could only copy {read} of {src_size} bytes")
