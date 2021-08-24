from __future__ import annotations

import collections.abc
import os
import pathlib
import typing
from typing import Any
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union

import fsspec
import pandas as pd

from pado.annotations import AnnotationProvider
from pado.annotations import Annotations
from pado.annotations import GroupedAnnotationProvider
from pado.images import GroupedImageProvider
from pado.images import Image
from pado.images import ImageId
from pado.images import ImageProvider
from pado.metadata import GroupedMetadataProvider
from pado.metadata import MetadataProvider
from pado.types import IOMode
from pado.types import UrlpathLike
from pado.io.files import urlpathlike_to_fsspec
from pado.io.files import urlpathlike_to_string
from pado.io.paths import get_root_dir
from pado.io.store import StoreType
from pado.io.store import get_store_type


class PadoDataset:
    __version__ = 2

    def __init__(self, urlpath: UrlpathLike, mode: IOMode = "r"):
        """open or create a new PadoDataset

        Parameters
        ----------
        urlpath:
            fsspec urlpath to `pado.dataset.toml` file, or its parent directory
        mode:
            'r' --> readonly, error if not there
            'r+' --> read/write, error if not there
            'a' --> read/write, create if not there, append if there
            'w' --> read/write, create if not there, truncate if there
            'x' --> read/write, create if not there, error if there

        """
        urlpath = get_root_dir(urlpath, allow_file="pado.dataset.toml")
        try:
            self._urlpath: str = urlpathlike_to_string(urlpath)
        except TypeError:
            raise
        if mode not in typing.get_args(IOMode):
            raise ValueError(f"unsupported mode {mode!r}")

        # file
        self._mode: IOMode = mode
        of = urlpathlike_to_fsspec(urlpath, mode=self._mode + "b")
        self._root: str = of.path
        self._fs: fsspec.AbstractFileSystem = of.fs

        # paths
        if not self.readonly:
            self._ensure_dir()

        # caches
        self._cached_index = None
        self._cached_image_provider = None
        self._cached_annotation_provider = None
        self._cached_metadata_provider = None

    @property
    def urlpath(self) -> str:
        """the urlpath pointing to the PadoDataset"""
        return self._urlpath

    @property
    def readonly(self) -> bool:
        """is the dataset in readonly mode"""
        return self._mode == "r"

    # === data properties ===

    @property
    def index(self) -> Sequence[ImageId]:
        """sequence of image_ids in the dataset"""
        if self._cached_index is None:
            image_ids = self.images.keys()
            if isinstance(image_ids, collections.abc.Sequence):
                self._cached_index = image_ids
            else:
                self._cached_index = list(image_ids)
        return self._cached_index

    @property
    def images(self) -> ImageProvider:
        """mapping image_ids to images in the dataset"""
        if self._cached_image_provider is None:

            providers = [
                ImageProvider.from_parquet(self._fs.open(p, mode='rb'))
                for p in self._fs.glob(self._get_fspath("*.image.parquet"))
                if self._fs.isfile(p)
            ]

            if len(providers) == 0:
                image_provider = ImageProvider()
            elif len(providers) == 1:
                image_provider = providers[0]
            else:
                image_provider = GroupedImageProvider(*providers)

            self._cached_image_provider = image_provider
        return self._cached_image_provider

    @property
    def annotations(self) -> AnnotationProvider:
        """mapping image_ids to annotations in the dataset"""
        if self._cached_annotation_provider is None:

            providers = [
                AnnotationProvider.from_parquet(self._fs.open(p, mode='rb'))
                for p in self._fs.glob(self._get_fspath("*.annotation.parquet"))
                if self._fs.isfile(p)
            ]

            if len(providers) == 0:
                annotation_provider = AnnotationProvider()
            elif len(providers) == 1:
                annotation_provider = providers[0]
            else:
                annotation_provider = GroupedAnnotationProvider(*providers)

            self._cached_annotation_provider = annotation_provider
        return self._cached_annotation_provider

    @property
    def metadata(self) -> MetadataProvider:
        """mapping image_ids to metadata in the dataset"""
        if self._cached_metadata_provider is None:

            providers = [
                MetadataProvider.from_parquet(self._fs.open(p, mode='rb'))
                for p in self._fs.glob(self._get_fspath("*.metadata.parquet"))
                if self._fs.isfile(p)
            ]

            if len(providers) == 0:
                metadata_provider = MetadataProvider()
            elif len(providers) == 1:
                metadata_provider = providers[0]
            else:
                metadata_provider = GroupedMetadataProvider(*providers)

            self._cached_metadata_provider = metadata_provider
        return self._cached_metadata_provider

    # === access ===

    def get_by_id(self, image_id: ImageId) -> PadoItem:
        return PadoItem(
            self.images.get(image_id),
            self.annotations.get(image_id),
            self.metadata.get(image_id),
        )

    def get_by_idx(self, idx: int) -> PadoItem:
        image_id = self.index[idx]
        return PadoItem(
            self.images.get(image_id),
            self.annotations.get(image_id),
            self.metadata.get(image_id),
        )

    # === data ingestion and summary ===

    def ingest_obj(self, obj: Any, *, identifier: Optional[str] = None) -> None:
        """ingest an object into the dataset"""
        if self.readonly:
            raise RuntimeError(f"{self!r} opened in readonly mode")

        if isinstance(obj, PadoDataset):
            for x in [obj.images, obj.metadata, obj.annotations]:
                self.ingest_obj(x)
        elif isinstance(obj, ImageProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for ImageProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.image.parquet")
            obj.to_parquet(self._fs.open(pth, mode="xb"))
            # invalidate caches
            self._cached_index = None
            self._cached_image_provider = None

        elif isinstance(obj, AnnotationProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for AnnotationProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.annotation.parquet")
            obj.to_parquet(self._fs.open(pth, mode="xb"))
            # invalidate caches
            self._cached_annotation_provider = None

        elif isinstance(obj, MetadataProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for MetadataProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.metadata.parquet")
            obj.to_parquet(self._fs.open(pth, mode="xb"))
            # invalidate caches
            self._cached_metadata_provider = None

        else:
            raise TypeError(f"unsupported object type {type(obj).__name__}: {obj!r}")

    def ingest_file(self, urlpath: UrlpathLike, *, identifier: Optional[str] = None) -> None:
        """ingest a file into the dataset"""
        if self.readonly:
            raise RuntimeError(f"{self!r} opened in readonly mode")
        store_type = get_store_type(urlpath)
        if store_type == StoreType.IMAGE:
            self.ingest_obj(ImageProvider.from_parquet(urlpath), identifier=identifier)

        else:
            raise NotImplementedError("todo: implement more files")

    # === internal utility methods ===

    def _get_fspath(self, *parts: Union[str, os.PathLike]) -> str:
        """return a fspath for a resource"""
        return os.fspath(pathlib.PurePath(self._root).joinpath(*parts))

    def _ensure_dir(self, *parts: Union[str, os.PathLike]) -> str:
        """ensure that a folder within the dataset exists"""
        pth = self._get_fspath(*parts)
        if not self._fs.isdir(pth):
            self._fs.mkdir(pth)
        return pth


# === helpers and utils =======================================================

class PadoItem(NamedTuple):
    image: Optional[Image]
    annotations: Optional[Annotations]
    metadata: Optional[pd.DataFrame]
