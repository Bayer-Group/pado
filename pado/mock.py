"""mock dataset for pado"""
from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from itertools import cycle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from fsspec.core import OpenFile
from shapely.geometry import Polygon

from pado.annotations import Annotation
from pado.annotations import AnnotationProvider
from pado.annotations import Annotations
from pado.annotations import AnnotationState
from pado.annotations import Annotator
from pado.annotations import AnnotatorType
from pado.annotations.providers import BaseAnnotationProvider
from pado.dataset import PadoDataset
from pado.images.ids import ImageId
from pado.images.image import Image
from pado.images.providers import BaseImageProvider
from pado.images.providers import ImageProvider
from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_fsspec
from pado.metadata import MetadataProvider
from pado.metadata.providers import BaseMetadataProvider
from pado.types import UrlpathLike

try:
    from tifffile import imwrite
except ImportError:  # pragma: no cover
    warnings.warn("must `pip install pado[mock]`")
    raise

__all__ = [
    "mock_dataset",
    "mock_annotations",
    "mock_metadata",
    "mock_images",
    "mock_image_ids",
    "temporary_mock_svs",
]


@contextmanager
def temporary_mock_svs(stem, size=(100, 100)) -> Iterator[Path]:
    """create a temporary tiff"""
    with TemporaryDirectory() as tmp_dir:
        size = (size[0], size[1], 3)
        data = np.random.randint(0, 255, size=size).astype(np.uint8)
        img_fn = Path(tmp_dir).expanduser().absolute() / f"{stem}.svs"
        svs_desc = (
            "Aperio Image Library pado.mock \r\n"
            "pretending to be an aperio svs... "
            "|AppMag = 40|MPP = 0.25"
        )
        # noinspection PyTypeChecker
        imwrite(img_fn, data, description=svs_desc)
        yield img_fn


def mock_image_ids(
    number: int, *, fmt="mock_image_{:d}.svs", site="mock"
) -> List[ImageId]:
    """create mock image_ids"""
    return [ImageId(fmt.format(idx), site=site) for idx in range(number)]


def mock_dataset(
    urlpath: Optional[UrlpathLike],
    *,
    num_images: int = 3,
    images_urlpath: Optional[UrlpathLike] = None,
    metadata_provider: bool = True,
    annotation_provider: bool = True,
) -> PadoDataset:
    """provide a mocked dataset for testing"""
    ds = PadoDataset(urlpath, mode="x")

    if images_urlpath is None:
        fs, path = urlpathlike_to_fs_and_path(ds.urlpath)
        _p = os.path.join(path, "_mocked_images")
        fs.mkdir(_p)
        images_urlpath = OpenFile(path=_p, fs=fs)

    ip = mock_images(images_urlpath, num_images)
    image_ids = list(ip.keys())
    ds.ingest_obj(ip)
    if metadata_provider:
        mp = mock_metadata(image_ids)
        ds.ingest_obj(mp)
    if annotation_provider:
        ap = mock_annotations(image_ids)
        ds.ingest_obj(ap)

    return ds


def mock_annotations(
    image_ids: Union[int, List[ImageId]],
    *,
    num_annotations: Iterable[int] = (5, 7, 9),
    base: bool = False,
) -> BaseAnnotationProvider:
    """return an annotation provider for testing"""
    # noinspection PyTypeChecker
    ap: BaseAnnotationProvider = {}
    if isinstance(image_ids, int):
        image_ids = mock_image_ids(image_ids)

    for image_id, num_anno in zip(image_ids, cycle(num_annotations)):
        ap[image_id] = Annotations.from_records(
            Annotation.from_obj(
                dict(
                    image_id=None,
                    identifier=None,
                    project="mock-project",
                    annotator=Annotator(
                        type=AnnotatorType.HUMAN,
                        name="Mock Mockington",
                    ),
                    state=AnnotationState.DONE,
                    classification=f"mocked-class-{dx}",
                    color="#ff00ff",
                    description="mocked description",
                    comment="none",
                    geometry=Polygon.from_bounds(
                        0 + dx * 32,
                        0 + dx * 32,
                        64 + dx * 32,
                        64 + dx * 32,
                    ),
                )
            ).to_record(image_id=image_id)
            for dx in range(num_anno)
        )

    if base:
        return ap
    else:
        return AnnotationProvider(ap)


def mock_metadata(
    image_ids: Union[int, List[ImageId]],
    *,
    num_findings: Iterable[int] = (1, 3, 5),
    base: bool = False,
) -> BaseMetadataProvider:
    """return a metadata provider for testing"""
    # noinspection PyTypeChecker
    mp: BaseMetadataProvider = {}
    if isinstance(image_ids, int):
        image_ids = mock_image_ids(image_ids)

    for image_id, num_f in zip(image_ids, cycle(num_findings)):
        # todo: provide richer metadata for testing
        mp[image_id] = pd.DataFrame(
            {
                "A": ["a"] * num_f,
                "B": [2] * num_f,
                "C": ["c"] * num_f,
                "D": [4] * num_f,
            },
            index=pd.Index([image_id.to_str()] * num_f),
        )

    if base:
        return mp
    else:
        return MetadataProvider(mp)


def mock_images(
    target: UrlpathLike, number: int, *, base: bool = False
) -> BaseImageProvider:
    """return an image provider for testing"""
    of = urlpathlike_to_fsspec(target)
    if not of.fs.isdir(of.path):
        raise NotADirectoryError(f"{target!r}")

    # noinspection PyTypeChecker
    ip: BaseImageProvider = {}
    fs, path = of.fs, of.path
    for image_id in mock_image_ids(number):

        stem, _ = os.path.splitext(image_id.last)
        with temporary_mock_svs(stem=stem, size=(512, 512)) as fn:
            img_data = fn.read_bytes()
            img_path = os.path.join(path, f"{stem}.svs")

            with fsopen(fs, img_path, mode="xb") as f:
                f.write(img_data)

        image = Image(
            fsopen(fs, img_path), load_metadata=True, load_file_info=True, checksum=True
        )
        ip[image_id] = image

    if base:
        return ip
    else:
        return ImageProvider(ip, identifier="mock")
