"""pado.types (and ABCs)

A collection of useful types and abstract base classes in pado
"""
from __future__ import annotations

import sys
from enum import Enum
from typing import IO
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

if sys.version_info >= (3, 11):
    from typing import Literal  # 3.8+
    from typing import NotRequired
    from typing import Protocol  # 3.8+
    from typing import TypedDict
    from typing import runtime_checkable
else:
    from typing_extensions import Literal
    from typing_extensions import NotRequired
    from typing_extensions import Protocol
    from typing_extensions import TypedDict
    from typing_extensions import runtime_checkable

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    import os

    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from fsspec import AbstractFileSystem

    from pado.annotations.annotation import Annotations
    from pado.images.ids import ImageId
    from pado.images.image import Image
    from pado.images.tiles import TileId


# --- types ---


@runtime_checkable
class OpenFileLike(Protocol):
    """minimal fsspec open file type"""

    fs: AbstractFileSystem
    path: str

    def __enter__(self) -> IO[bytes]:
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


UrlpathLike = Union[str, "os.PathLike[str]", OpenFileLike]
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
    ) -> Iterator[Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        ...


PI = TypeVar("PI", bound="SerializableItem")


@runtime_checkable
class SerializableItem(Protocol):
    __fields__: tuple[str, ...]

    @classmethod
    def from_obj(cls: Type[PI], obj: Any) -> PI:
        ...

    def to_record(
        self, image_id: ImageId | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        ...


class FilterMissing(str, Enum):
    WARN = "warn"
    ERROR = "error"
    IGNORE = "ignore"


class CollatedPadoItems(TypedDict):
    """collected batch dictionary used in pado.itertools"""

    id: list[ImageId]
    image: list[Image]
    metadata: list[pd.DataFrame]
    annotations: list[Annotations]


class CollatedPadoTileItems(TypedDict):
    """collected batch dictionary used in pado.itertools"""

    id: list[TileId]
    tile: Union[list[npt.ArrayLike], npt.ArrayLike]
    metadata: list[pd.DataFrame]
    annotations: list[Annotations]


class SerializedImageId(TypedDict):
    """a json serialized image id"""

    image_id: Sequence[str]
    site: NotRequired[str]
