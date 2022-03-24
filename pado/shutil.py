from __future__ import annotations

from pathlib import PurePath
from typing import Callable

from pado.dataset import PadoDataset
from pado.io.files import urlpathlike_local_via_fs


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

    def _transfer_resources(p, destination, *, chunked=False):
        if not destination or not hasattr(p, "df"):
            return
        fs0 = ds0._fs
        fs1, get_fspath1 = ds1._fs, ds1._get_fspath
        dst_dir = get_fspath1(destination)
        if not fs1.isdir(dst_dir):
            fs1.mkdir(dst_dir, create_parents=True)
        for up in p.df.urlpath:
            of0 = urlpathlike_local_via_fs(up, fs0)
            name = PurePath(of0.path).name
            path_remote = get_fspath1(destination, name)
            if fs1.isfile(path_remote) and fs1.size(path_remote) > 0:
                progress_callback(f"EXISTS {name}")
                continue
            else:
                of1 = fs1.open(path_remote, mode="wb")

            try:
                with of0 as f0, of1 as f1:
                    progress_callback(name)
                    if chunked:
                        f0_read = f0.read
                        f1_write = f1.write
                        bufsize = fs1.blocksize
                        while True:
                            progress_callback(".")
                            buf = f0_read(bufsize)
                            if not buf:
                                break
                            f1_write(buf)
                    else:
                        buf = f0.read()
                        progress_callback(". received")
                        f1.write(buf)
                        progress_callback(". transferred")
            except FileNotFoundError:
                progress_callback(f"NOT FOUND {name}")
                continue

    def _transfer(p):
        if keep_individual_providers and hasattr(p, "providers"):
            for _p in p.providers:
                progress_callback(repr(_p))
                try:
                    ds1.ingest_obj(_p)
                except FileExistsError:
                    pass
        else:
            progress_callback(repr(p))
            try:
                ds1.ingest_obj(p)
            except FileExistsError:
                pass

    if image_providers:
        _transfer(ds0.images)
        _transfer_resources(ds0.images, images_path)
    if metadata_providers:
        _transfer(ds0.metadata)
    if annotation_providers:
        _transfer(ds0.annotations)
    if image_prediction_providers:
        _transfer(ds0.predictions.images)
        _transfer_resources(ds0.predictions.images, image_predictions_path)
