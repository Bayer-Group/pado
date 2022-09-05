from __future__ import annotations

import base64
import enum
import hashlib
import itertools
import json
from collections import deque
from importlib import import_module
from typing import Any
from typing import BinaryIO
from typing import Container
from typing import NamedTuple

from pado.io.files import urlpathlike_to_fs_and_path
from pado.types import UrlpathLike

__all__ = [
    "Algorithm",
    "Checksum",
    "compute_checksum",
    "verify_checksum",
]


class Algorithm(str, enum.Enum):
    MD5 = "md5"
    AWS_ETAG = "aws-etag"
    CRC32C = "crc32c"
    GCS_ETAG = "gcs-etag"


Algorithm.ALL_SUPPORTED = frozenset(
    [Algorithm.MD5, Algorithm.AWS_ETAG, Algorithm.CRC32C]
)
Algorithm.NON_ETAG = frozenset([Algorithm.MD5, Algorithm.CRC32C])


class Checksum(NamedTuple):
    algorithm: Algorithm
    file_size: int | None
    value: str

    def __str__(self):
        algorithm = self.algorithm.value
        file_size = json.dumps(self.file_size)
        return f"{algorithm}:{file_size}:{self.value}"

    @classmethod
    def from_str(cls, string) -> Checksum | tuple[Checksum, ...]:
        checksums = []
        for string_i in string.split("::"):
            algorithm, json_file_size, value = string_i.split(":")
            file_size = json.loads(json_file_size)
            checksums.append(cls(Algorithm(algorithm), file_size, value))
        if len(checksums) == 0:
            raise ValueError("empty checksum string")
        elif len(checksums) == 1:
            return checksums[0]
        else:
            return tuple(checksums)

    @classmethod
    def join_checksums(cls, checksums: tuple[Checksum]) -> str | None:
        return "::".join(map(str, checksums)) or None


def ensure_str(x: bytes | str) -> str:
    if isinstance(x, bytes):
        return x.decode()
    else:
        return x


def _get_md5(x: bytes):
    # md5 is only used for file integrity checks
    return hashlib.md5(x)  # nosec B303


def checksum_multiple(
    f: BinaryIO, *, algorithms: Container[Algorithm], chunk_size: int
) -> tuple[Checksum]:
    global _CRC32C_Checksum
    if not algorithms:
        raise ValueError("must provide at least one algorithm")

    build = {}
    if Algorithm.MD5 in algorithms:
        build[Algorithm.MD5] = _get_md5()
    if Algorithm.CRC32C in algorithms:
        if _CRC32C_Checksum is None:
            try:
                _CRC32C_Checksum = getattr(import_module("google_crc32c"), "Checksum")
            except ImportError as err:
                raise RuntimeError("crc32c requires `google-crc32c` package") from err

        build[Algorithm.CRC32C] = _CRC32C_Checksum()
    if Algorithm.AWS_ETAG in algorithms:
        build[Algorithm.AWS_ETAG] = _AWSETagChecksum(block_size=10 * 1024 * 1024)
    if Algorithm.GCS_ETAG in algorithms:
        raise NotImplementedError("...")

    ms = list(build.values())
    length = 0
    for data in iter(lambda: f.read(chunk_size), b""):
        length += len(data)
        for m in ms:
            m.update(data)

    return tuple(
        (Checksum(key, length, ensure_str(m.hexdigest())) for key, m in build.items())
    )


def checksum_md5(f: BinaryIO, *, chunk_size: int = 10 * 1024 * 1024) -> str:
    """return the md5sum"""
    m = _get_md5()
    for data in iter(lambda: f.read(chunk_size), b""):
        m.update(data)
    return m.hexdigest()


class _AWSETagChecksum:
    def __init__(self, block_size: int):
        self._blocks = []
        self._block_size = int(block_size)
        self._queue = deque()

    def update(self, data: bytes):
        self._queue.append(data)
        self._digest_block()

    def _digest_block(self, final: bool = False) -> None:
        if not self._queue:
            return
        # get all data from queue
        data = b"".join(self._queue)
        self._queue.clear()

        # check if we should digest
        if len(data) >= (self._block_size if not final else 1):
            # feed the input in blocks
            inp = data[: self._block_size]
            self._blocks.append(_get_md5(inp))
            # requeue the remaining data
            rem = data[self._block_size :]
            if rem:
                self._queue.append(rem)
            # rerun in case we feed more than one block
            if self._queue:
                return self._digest_block(final=final)
        return

    def hexdigest(self):
        self._digest_block(final=True)

        if len(self._blocks) == 0:
            etag = _get_md5().hexdigest()
            return f'"{etag}"'
        if len(self._blocks) == 1:
            etag = self._blocks[0].hexdigest()
            return f'"{etag}"'
        else:
            _concatenated_md5s = b"".join(m.digest() for m in self._blocks)
            etag = _get_md5(_concatenated_md5s).hexdigest()
            return f'"{etag}-{len(self._blocks)}"'


def checksum_aws_etag(f: BinaryIO, *, chunk_size: int = 10 * 1024 * 1024) -> str:
    """returns the aws etag checksum

    md5 for single chunk, md5 of concatenated chunk md5s for multi chunk.
    """
    chunk_md5s = [_get_md5(data) for data in iter(lambda: f.read(chunk_size), b"")]
    num_chunks = len(chunk_md5s)

    if num_chunks == 0:
        etag = _get_md5().hexdigest()
        return f'"{etag}"'

    if num_chunks == 1:
        etag = chunk_md5s[0].hexdigest()
        return f'"{etag}"'

    else:
        _concatenated_md5s = b"".join(m.digest() for m in chunk_md5s)
        etag = _get_md5(_concatenated_md5s).hexdigest()
        return f'"{etag}-{num_chunks}"'


# noinspection PyUnusedLocal
def checksum_gcs_etag(f: BinaryIO, *, chunk_size: int = 10 * 1024 * 1024) -> str:
    """returns the gcs etag checksum"""
    raise NotImplementedError("todo...")


_CRC32C_Checksum = None


def checksum_crc32c(f: BinaryIO, *, chunk_size: int = 10 * 1024 * 1024) -> str:
    """returns the gcs crc32c checksum"""
    global _CRC32C_Checksum

    if _CRC32C_Checksum is None:
        try:
            _CRC32C_Checksum = getattr(import_module("google_crc32c"), "Checksum")
        except ImportError as err:
            raise RuntimeError("crc32c requires `google-crc32c` package") from err

    _chk = _CRC32C_Checksum()
    deque(_chk.consume(f.read, chunk_size), maxlen=0)
    return _chk.hexdigest()


def _convert_checksum(
    checksum: Checksum,
    *,
    algorithm: Algorithm,
    block_size: int = -1,
) -> Checksum:
    """try converting to a different algorithm without computing"""
    if checksum.algorithm == algorithm:
        return checksum

    alg, size, chk = checksum

    # md5 to aws-etag
    if alg == Algorithm.MD5 and algorithm == Algorithm.AWS_ETAG and size < block_size:
        return Checksum(algorithm, size, f'"{chk}"')

    # aws-etag to md5
    if alg == Algorithm.AWS_ETAG and algorithm == Algorithm.MD5 and "-" not in chk:
        return Checksum(algorithm, size, chk.strip('"'))

    raise ValueError(f"can't convert {checksum!r} to {algorithm!r}")


def get_precomputed_checksums(
    urlpath_or_meta: UrlpathLike | dict[str, Any],
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[Checksum, ...]:
    """return a list of checksums for the urlpath

    tries to return checksums precomputed by the filesystem
    provider from the metadata returned by fs.info()

    """
    if isinstance(urlpath_or_meta, dict):
        meta = urlpath_or_meta
    else:
        fs, path = urlpathlike_to_fs_and_path(
            urlpath_or_meta, storage_options=storage_options
        )
        meta = fs.info(path)

    _checksums = []

    if "size" in meta:
        size = meta["size"]
    elif "Size" in meta:
        size = meta["Size"]
    else:
        size = -1

    if "md5Hash" in meta:
        # google cloud
        md5 = base64.b64decode(meta["md5Hash"]).hex()
        _checksums.append(Checksum(Algorithm.MD5, size, md5))

    if "crc32c" in meta:
        crc32 = base64.b64decode(meta["crc32c"]).hex()
        _checksums.append(Checksum(Algorithm.CRC32C, size, crc32))

    if "etag" in meta:
        # has etag (gcp)
        etag = meta["etag"]
        _checksums.append(Checksum(Algorithm.GCS_ETAG, size, etag))

    if "ETag" in meta:
        # has etag (aws)
        etag = meta["ETag"]
        if etag[0] != '"' or etag[-1] != '"':
            raise AssertionError(f"unexpected aws ETag without quotes: {etag!r}")
        else:
            _checksums.append(Checksum(Algorithm.AWS_ETAG, size, etag[1:-1]))
    return tuple(_checksums)


def compute_checksum(
    f: UrlpathLike,
    *,
    algorithms: Algorithm | set[Algorithm] = Algorithm.NON_ETAG,
    available_only: bool = False,
    raise_not_local: bool = True,
    storage_options: dict[str, Any] | None = None,
) -> tuple[Checksum] | str:
    """compute a checksum trying to take advantage of remote fs

    Parameters
    ----------
    f:
        urlpath-like path / file / binaryIO object
    algorithms:
        checksum algorithms to be computed. default md5 and crc32c
    available_only:
        don't compute, and return only what's available through the fs
    raise_not_local:
        raise if a checksum would need to be computed by reading from a remote
    storage_options:
        options passed through with urlpathlike

    Returns
    -------
    checksum:
        the Checksum

    """
    fs, path = urlpathlike_to_fs_and_path(f, storage_options=storage_options)

    meta = fs.info(path)
    checksums = get_precomputed_checksums(meta)

    if not algorithms:
        raise ValueError("must provide algorithms")
    elif isinstance(algorithms, Algorithm):
        algorithms = {algorithms}
    else:
        algorithms = algorithms

    precomputed_algorithms: dict[Algorithm, Checksum] = {
        c.algorithm: c for c in checksums
    }
    if algorithms.issubset(precomputed_algorithms):
        return checksums

    missing_algorithms = algorithms - set(precomputed_algorithms)
    converted_checksums = {}
    for a in missing_algorithms:
        for c in checksums:
            try:
                converted_checksums[a] = _convert_checksum(c, algorithm=a)
            except ValueError:
                pass
    missing_algorithms -= set(converted_checksums)
    precomputed_algorithms.update(converted_checksums)
    if not missing_algorithms or available_only:
        return tuple(precomputed_algorithms.values())

    if raise_not_local and not getattr(fs, "local_file", False):
        raise ValueError(f"won't checksum non-local: {f!r}")

    with fs.open(path, mode="rb") as f:
        checksums = checksum_multiple(
            f, algorithms=missing_algorithms, chunk_size=2**21
        )

    return checksums


def _to_checksums(x: tuple[Checksum] | Checksum | str) -> tuple[Checksum]:
    if isinstance(x, str):
        x = Checksum.from_str(x)
    if isinstance(x, Checksum):
        return (x,)
    elif isinstance(x, tuple) and x and all(isinstance(y, Checksum) for y in x):
        return x
    else:
        raise ValueError(
            f"expected str, Checksum or tuple[Checksum], got {type(x).__name__}"
        )


def compare_checksums(
    test: tuple[Checksum] | Checksum | str,
    target: tuple[Checksum] | Checksum | str,
) -> bool:
    """compare checksums. raises ValueError if not possible"""
    test = _to_checksums(test)
    target = _to_checksums(target)
    for c0, c1 in itertools.product(test, target):
        if c0.algorithm == c1.algorithm:
            return c0 == c1
        try:
            c0 = _convert_checksum(c0, algorithm=c1.algorithm)
        except ValueError:
            pass
        else:
            return c0 == c1
    else:
        raise ValueError("could not compare checksums: {test!r} and {target!r}")


def verify_checksum(
    test: UrlpathLike,
    *,
    target: Checksum,
    raise_not_local: bool = True,
    storage_options: dict[str, Any] | None = None,
) -> tuple[bool, tuple[Checksum]]:
    """verify if checksums match"""
    fs, path = urlpathlike_to_fs_and_path(test, storage_options=storage_options)
    meta = fs.info(path)
    checksums = get_precomputed_checksums(meta)

    try:
        # test if one of the available checksums is a match
        return compare_checksums(checksums, target), checksums
    except ValueError:
        pass

    if raise_not_local and not getattr(fs, "local_file", False):
        raise ValueError(f"won't checksum non-local: {test!r}")

    # fixme: ...
    raise NotImplementedError("todo compute all checksums")
