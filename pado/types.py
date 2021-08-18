"""pado.types (and ABCs)

A collection of useful types and abstract base classes in pado
"""
from __future__ import annotations

import sys
from typing import AnyStr
from typing import ContextManager
from typing import IO
from typing import TYPE_CHECKING
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

    from fsspec import AbstractFileSystem


# --- types ---

@runtime_checkable
class OpenFileLike(Protocol, ContextManager[IO[AnyStr]]):
    """minimal fsspec open file type"""
    fs: AbstractFileSystem
    path: str
    def __enter__(self) -> IO[AnyStr]: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...


UrlpathLike = Union[AnyStr, "os.PathLike[AnyStr]", OpenFileLike[AnyStr]]
IOMode = Literal['r', 'r+', 'w', 'a', 'x']
