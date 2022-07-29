"""pado.types (and ABCs)

A collection of useful types and abstract base classes in pado
"""
from __future__ import annotations

import sys
from typing import IO
from typing import TYPE_CHECKING
from typing import Any
from typing import AnyStr
from typing import ContextManager
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

if sys.version_info >= (3, 8):
    from typing import Literal  # 3.8+
    from typing import Protocol  # 3.8+
    from typing import runtime_checkable
else:
    from typing_extensions import Literal
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    import os

    import numpy.typing as npt
    from fsspec import AbstractFileSystem

    from pado.images import ImageId


# --- types ---


@runtime_checkable
class OpenFileLike(Protocol, ContextManager[IO[AnyStr]]):
    """minimal fsspec open file type"""

    fs: AbstractFileSystem
    path: str

    def __enter__(self) -> IO[AnyStr]:
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


UrlpathLike = Union[AnyStr, "os.PathLike[AnyStr]", OpenFileLike[AnyStr]]
IOMode = Literal["r", "r+", "w", "a", "x"]
FsspecIOMode = Literal["r", "rb", "w", "wb", "a", "ab", "x", "xb"]


class UrlpathWithStorageOptions(NamedTuple):
    """container for urlpath with optional storage options"""

    urlpath: UrlpathLike
    storage_options: dict[str, str | int | float] | None = None


@runtime_checkable
class DatasetSplitter(Protocol):
    """splitter classes from sklearn.model_selection"""

    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]],
        groups: Optional[Sequence[Any]],
    ) -> Iterator[Tuple[npt.NDArray[int], npt.NDArray[int]]]:
        ...


PI = TypeVar("PI", bound="SerializableItem")


@runtime_checkable
class SerializableItem(Protocol):
    __fields__: tuple[str, ...]

    @classmethod
    def from_obj(cls: Type[PI], obj: Any) -> PI:
        ...

    def to_record(self, image_id: ImageId | None = None) -> dict[str, Any]:
        ...
