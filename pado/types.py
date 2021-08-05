"""pado.types (and ABCs)

A collection of useful types and abstract base classes in pado
"""
from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod
from typing import AnyStr
from typing import IO
from typing import Mapping
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import pandas as pd

if sys.version_info >= (3, 8):
    from typing import Literal  # 3.8+
    from typing import Protocol  # 3.8+
    from typing import runtime_checkable
else:
    from typing_extensions import Literal
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable

if TYPE_CHECKING:
    import os

    from fsspec import AbstractFileSystem

    from pado.annotations import Annotations
    from pado.images import ImageId
    from pado.images import Image


# --- types ---

T = TypeVar("T", str, bytes)


@runtime_checkable
class OpenFileLike(Protocol[T]):
    """minimal fsspec open file type"""
    fs: AbstractFileSystem
    path: str
    def __enter__(self) -> IO[T]: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...


UrlpathLike = Union[AnyStr, "os.PathLike[AnyStr]", OpenFileLike[AnyStr]]
IOMode = Literal['r', 'r+', 'w', 'a', 'x']


# --- abcs ---

class DatasetABC(ABC):

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        raise NotImplementedError("implement in subclass")

    @property
    @abstractmethod
    def images(self) -> Mapping[ImageId, Image]:
        raise NotImplementedError("implement in subclass")

    @property
    @abstractmethod
    def annotations(self) -> Mapping[ImageId, Annotations]:
        raise NotImplementedError("implement in subclass")
