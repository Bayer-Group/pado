"""pado.create: tools to create pado datasets

This module provides functions for creating the various providers used
in pado datasets.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import uuid
import warnings
from contextlib import nullcontext
from functools import partial
from itertools import repeat
from typing import Any
from typing import Container
from typing import Mapping

import fsspec.asyn
from rich.progress import Progress

from pado.collections import is_valid_identifier
from pado.images.ids import GetImageIdFunc
from pado.images.ids import ImageId
from pado.images.ids import image_id_from_parts
from pado.images.image import Image
from pado.images.providers import ImageProvider
from pado.io.files import find_files
from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_fs_and_path
from pado.types import UrlpathLike

try:
    import rich
    import rich.console
    from rich.progress import track
except ImportError:
    rich = None
    track = None


__all__ = [
    "create_image_provider",
]


def _subprocess_init():
    """fix fsspec hanging on subprocesses"""
    # https://github.com/fsspec/filesystem_spec/pull/963/files
    fsspec.asyn.reset_lock()


def _star_call(fn, args):
    """helper to use with multiprocessing.Pool.imap"""
    return fn(*args)


def _open_image_from_file_and_parts(
    of_parts: tuple,
    identifier: str | None,
    image_id_func: GetImageIdFunc,
    available_ids: Container[ImageId],
    checksum: bool | Mapping[ImageId, str],
) -> tuple[ImageId | None, Image | None, BaseException | None]:
    """helper for getting images from files"""
    of, parts = of_parts
    # get image_id from file
    try:
        image_id = image_id_func(of, parts, identifier)
    except KeyboardInterrupt:
        raise
    except BaseException as err:
        return None, None, err

    # return early
    if image_id is None:
        return None, None, None
    elif image_id in available_ids:
        return image_id, None, None

    # get checksum
    chk: bool | str
    if isinstance(checksum, bool):
        chk = checksum
    else:
        chk = checksum[image_id]

    # get image
    try:
        image = Image(
            of,
            load_metadata=True,
            load_file_info=True,
            checksum=chk,
        )
    except KeyboardInterrupt:
        raise
    except BaseException as err:
        return image_id, None, err
    else:
        return image_id, image, None


def create_image_provider(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    output_urlpath: UrlpathLike | None,
    identifier: str | None = None,
    checksum: bool | Mapping[ImageId, str] = True,
    resume: bool = False,
    ignore_broken: bool = True,
    image_id_func: GetImageIdFunc = image_id_from_parts,
    progress: bool = False,
    workers: int = 0,
    search_storage_options: dict[str, Any] | None = None,
    output_storage_options: dict[str, Any] | None = None,
) -> ImageProvider:
    """create an image provider from a directory containing images

    Allows you to create a pado ImageProvider instance from a directory
    containing compatible images. All images that are found using the
    glob pattern are integrated into the provider.

    Parameters
    ----------
    search_urlpath:
        The search directory. Can be a local dir `/my/path` or a remote
        location `s3://somebucket/path`.
    search_glob:
        The glob pattern for files to be integrated, i.e. "**/*.svs".
    output_urlpath:
        The output location for the image provider. Can be any writable
        urlpath, i.e. `/somewhere/here/` or `None` for in memory store.
    identifier:
        The str identifier for the provider. Should start and end with
        r"[a-zA-Z0-9_]" and only contain r"[a-zA-Z0-9_-]*".
    checksum:
        If `True`, will checksum each image and store the checksum in
        the provider. Disable with `False`. If a mapping from `ImageId`
        to `str` is provided, each image will be checksum-ed and an
        error will be raised in case there's a mismatch.
    resume:
        If `True` allows to append to an existing provider stored at
        `output_urlpath`.
    ignore_broken:
        Toggles if broken images should be ignored.
    image_id_func:
        A user provider function matching the `GetImageIdFunc` signature
        that takes information about the image and returns an `ImageId`.
    progress:
        Toggles is progress information should be printed to console.
    workers:
        Set the number of worker processes for building the provider.
        0 means use the current process only.
    search_storage_options:
        Optional storage options for the `search_urlpath`.
    output_storage_options:
        Optional storage options for the `output_urlpath`.

    """
    if progress and rich is None:
        warnings.warn(
            "progress=True requires `rich`. Install via `pip install rich`",
            stacklevel=2,
        )
        progress = False

    if resume:
        try:
            ip = ImageProvider.from_parquet(urlpath=output_urlpath)
        except FileNotFoundError:
            ip = ImageProvider(identifier=identifier)
        else:
            if identifier is None:
                identifier = ip.identifier
    else:
        ip = ImageProvider(identifier=identifier)

    if identifier is None:
        identifier = str(uuid.uuid4())
    if is_valid_identifier(identifier):
        pass
    else:
        raise ValueError(f"not a valid identifier: {identifier!r}")

    if progress and rich:
        console = rich.get_console()
        spinner = console.status("[bold green]Searching files...")
    else:
        spinner = nullcontext()

    with spinner:
        files_and_parts = find_files(
            search_urlpath,
            glob=search_glob,
            storage_options=search_storage_options,
        )

    if workers > 0:
        _pool = multiprocessing.Pool(
            processes=workers,
            initializer=_subprocess_init,
        )
    else:
        e = type("_executor", (), {"imap_unordered": map})
        _pool = nullcontext(e)

    broken = []

    if progress:
        _progress = Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TimeRemainingColumn(),
        )
    else:
        _progress = nullcontext(None)

    with _progress as _p:
        if _p:
            task_id = _p.add_task(
                "[cyan]Gathering Images...", total=len(files_and_parts)
            )
        else:
            task_id = None

        try:
            with _pool as pool:
                iid_img_err = pool.imap_unordered(
                    partial(_star_call, _open_image_from_file_and_parts),
                    zip(
                        files_and_parts,
                        repeat(identifier),
                        repeat(image_id_func),
                        repeat(set(ip)),
                        repeat(checksum),
                    ),
                )

                for image_id, image, err in iid_img_err:
                    # catch errors
                    if err is not None:
                        if image_id is None:
                            raise err
                        elif not ignore_broken:
                            raise RuntimeError(f"{image_id!r}") from err
                        else:
                            if _p:
                                _p.console.print("[red]{image_id!s}: {err!r}")
                            broken.append((image_id, err))
                    else:
                        if image_id is not None:
                            ip[image_id] = image
                    if _p:
                        _p.advance(task_id)

        finally:
            if output_urlpath is not None:
                if progress and rich:
                    console = rich.console.Console(stderr=True)
                    console.print(
                        f"[yellow]storing ImageProvider at {output_urlpath!r}"
                    )
                elif progress:
                    print(
                        f"Storing ImageProvider at {output_urlpath!r}", file=sys.stderr
                    )

                fs, pth = urlpathlike_to_fs_and_path(
                    output_urlpath, storage_options=output_storage_options
                )
                if fs.isdir(pth):
                    pth = os.path.join(pth, f"{identifier}.image.parquet")
                    output_urlpath = fsopen(fs, pth, mode="wb")
                ip.to_parquet(output_urlpath)
            if broken:
                if rich:
                    console = rich.console.Console(stderr=True)
                    for image_id, err in broken:
                        console.print(f"[red] {image_id!s}: {err!r}")
                else:
                    print(f"ERROR: {image_id!s}: {err!r}", file=sys.stderr)

    return ip
