from __future__ import annotations

import os.path
from contextlib import ExitStack
from functools import partial
from typing import Callable

from fsspec import AbstractFileSystem

from pado.dataset import PadoDataset
from pado.images.providers import update_image_provider_urlpaths
from pado.io.files import urlpathlike_local_via_fs
from pado.predictions.providers import ImagePredictionProvider


def transfer(
    ds0: PadoDataset,
    ds1: PadoDataset,
    *,
    image_providers: bool = False,
    metadata_providers: bool = False,
    annotation_providers: bool = False,
    image_prediction_providers: bool = False,
    images_path: str | None = None,
    image_predictions_path: str | None = None,
    keep_individual_providers: bool = True,
    progress_callback: Callable[[str], ...] = lambda _: None,
) -> None:
    """move dataset contents from one to another"""
    if not isinstance(ds0, PadoDataset):
        raise TypeError(f"ds0={ds0!r} not a PadoDataset")
    if not isinstance(ds1, PadoDataset):
        raise TypeError(f"ds1={ds1!r} not a PadoDataset")
    elif ds1.readonly:
        raise ValueError(f"ds1 must be writable: {ds1!r}")

    # noinspection PyProtectedMember
    cp = partial(
        _transfer,
        ds0=ds0,
        ds1=ds1,
        progress_callback=progress_callback,
        keep_individual_providers=keep_individual_providers,
    )

    if image_providers:
        cp(ds0.images, item_urlpath_destination=images_path)
    if metadata_providers:
        cp(ds0.metadata)
    if annotation_providers:
        cp(ds0.annotations)
    if image_prediction_providers:
        cp(ds0.predictions.images, item_urlpath_destination=image_predictions_path)


def _transfer(
    p,
    *,
    ds0: PadoDataset,
    ds1: PadoDataset,
    progress_callback,
    keep_individual_providers,
    item_urlpath_destination=None,
    chunked=False,
) -> None:

    # noinspection PyProtectedMember
    fs0: AbstractFileSystem = ds0._fs
    # noinspection PyProtectedMember
    fs1: AbstractFileSystem = ds1._fs
    # noinspection PyProtectedMember
    get_fspath1: Callable[[str], str] = ds1._get_fspath

    if keep_individual_providers and hasattr(p, "providers"):
        providers = p.providers
    else:
        providers = [p]

    for _p in providers:
        progress_callback(repr(_p))

        # copy items referenced by urlpath in the provider if requested
        if item_urlpath_destination is not None and hasattr(_p, "df"):

            # create destination dir
            dst_dir = get_fspath1(item_urlpath_destination)
            if not fs1.isdir(dst_dir):
                fs1.mkdir(dst_dir, create_parents=True)

            for item_urlpath_src in _p.df.urlpath:
                # we need to see access locally referenced items remotely,
                # if we access the dataset remotely
                of0 = urlpathlike_local_via_fs(item_urlpath_src, fs0)

                # create the remote path
                _, name = os.path.split(of0.path)
                item_urlpath_dst = get_fspath1(item_urlpath_destination, name)

                # skip if remote exists
                if fs1.isfile(item_urlpath_dst) and fs1.size(item_urlpath_dst) > 0:
                    progress_callback(f". EXISTS {name}")
                    continue
                else:
                    of1 = fs1.open(item_urlpath_dst, mode="wb")

                # transfer file
                with ExitStack() as stack:
                    try:
                        f0 = stack.enter_context(of0)
                    except FileNotFoundError:
                        progress_callback(f"NOT FOUND {name}")
                        continue
                    else:
                        progress_callback(name)
                        f1 = stack.enter_context(of1)

                    if chunked:
                        f0_read = f0.read
                        f1_write = f1.write
                        buf_size = fs1.blocksize
                        while True:
                            progress_callback(".")
                            buf = f0_read(buf_size)
                            if not buf:
                                break
                            f1_write(buf)
                    else:
                        buf = f0.read()
                        progress_callback(". received")
                        f1.write(buf)
                        progress_callback(". transferred")

            # re-associate the uploaded provider's item urlpaths
            _p = update_image_provider_urlpaths(
                fs1.open(get_fspath1(item_urlpath_destination)),
                search_glob="*.*",
                provider=_p,
                inplace=False,
                ignore_ambiguous=True,
                progress=True,
                provider_cls=type(_p),
            )

        try:
            ds1.ingest_obj(_p)
        except FileExistsError:
            progress_callback(". PROVIDER EXISTED ALREADY, OVERWRITING")
            if not isinstance(_p, ImagePredictionProvider):
                raise RuntimeError("provider not an image provider")
            ds1.ingest_obj(_p, overwrite=True)
