from __future__ import annotations

import collections.abc
import os
import pathlib
import uuid
from typing import Any
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from typing import get_args

import fsspec
import numpy as np
import pandas as pd

from pado.annotations import AnnotationProvider
from pado.annotations import Annotations
from pado.annotations import GroupedAnnotationProvider
from pado.images import GroupedImageProvider
from pado.images import Image
from pado.images import ImageId
from pado.images import ImageProvider
from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_string
from pado.io.paths import get_root_dir
from pado.io.store import StoreType
from pado.io.store import get_store_type
from pado.metadata import GroupedMetadataProvider
from pado.metadata import MetadataProvider
from pado.types import DatasetSplitter
from pado.types import IOMode
from pado.types import UrlpathLike

__all__ = ["PadoDataset"]


class PadoDataset:
    __version__ = 2

    def __init__(self, urlpath: UrlpathLike | None, mode: IOMode = "r"):
        """open or create a new PadoDataset

        Parameters
        ----------
        urlpath:
            fsspec urlpath to `pado.dataset.toml` file, or its parent directory.
            If explicitly set to None, uses an in-memory filesystem to store the dataset.
        mode:
            'r' --> readonly, error if not there
            'r+' --> read/write, error if not there
            'a' --> read/write, create if not there, append if there
            'w' --> read/write, create if not there, truncate if there
            'x' --> read/write, create if not there, error if there

        """
        if urlpath is None:
            # enable in-memory pado datasets
            urlpath = f"memory://pado-{uuid.uuid4()}"

        urlpath = get_root_dir(urlpath, allow_file="pado.dataset.toml")
        try:
            self._urlpath: str = urlpathlike_to_string(urlpath)
        except TypeError as err:
            raise TypeError(f"incompatible urlpath {urlpath!r}") from err
        if mode not in get_args(IOMode):
            raise ValueError(f"unsupported mode {mode!r}")

        # file
        self._mode: IOMode = mode

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
    def _fs(self) -> fsspec.AbstractFileSystem:
        fs, _ = urlpathlike_to_fs_and_path(self._urlpath)
        return fs

    @property
    def _root(self) -> str:
        _, path = urlpathlike_to_fs_and_path(self._urlpath)
        return path

    @property
    def readonly(self) -> bool:
        """is the dataset in readonly mode"""
        return self._mode == "r"

    def __repr__(self):
        return f"{type(self).__name__}({self.urlpath!r}, mode={self._mode!r})"

    # === data properties ===

    @property
    def index(self) -> Sequence[ImageId]:
        """sequence of image_ids in the dataset"""
        if self._cached_index is None:
            image_ids = self.images.keys()
            if isinstance(image_ids, collections.abc.Sequence):
                self._cached_index = image_ids
            else:
                self._cached_index = tuple(image_ids)
        return self._cached_index

    @property
    def images(self) -> ImageProvider:
        """mapping image_ids to images in the dataset"""
        if self._cached_image_provider is None:

            fs = self._fs
            providers = [
                ImageProvider.from_parquet(fsopen(fs, p, mode='rb'))
                for p in fs.glob(self._get_fspath("*.image.parquet"))
                if fs.isfile(p)
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

            fs = self._fs
            providers = [
                AnnotationProvider.from_parquet(fsopen(fs, p, mode='rb'))
                for p in fs.glob(self._get_fspath("*.annotation.parquet"))
                if fs.isfile(p)
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

            fs = self._fs
            providers = [
                MetadataProvider.from_parquet(fsopen(fs, p, mode='rb'))
                for p in fs.glob(self._get_fspath("*.metadata.parquet"))
                if fs.isfile(p)
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
            image_id,
            self.images.get(image_id),
            self.annotations.get(image_id),
            self.metadata.get(image_id),
        )

    def get_by_idx(self, idx: int) -> PadoItem:
        image_id = self.index[idx]
        return PadoItem(
            image_id,
            self.images.get(image_id),
            self.annotations.get(image_id),
            self.metadata.get(image_id),
        )

    # === filter functionality ===

    def filter(
        self,
        ids_or_func: Sequence[ImageId] | Callable[[PadoItem], bool],
        *,
        urlpath: Optional[UrlpathLike] = None
    ) -> PadoDataset:
        """filter a pado dataset

        Parameters
        ----------
        ids_or_func:
            either a Sequence of ImageId instances or a function that gets
            called with each PadoItem and returns a bool indicating if it should
            be kept or not.
        urlpath:
            a urlpath to store the filtered provider. If None (default) returns
            a in-memory PadoDataset

        """
        # todo: if this is not fast enough might consider lazy filtering

        if isinstance(ids_or_func, ImageId):
            raise ValueError("must provide a list of ImageIds")

        if isinstance(ids_or_func, Sequence):
            ids = pd.Series(ids_or_func).apply(str.__call__)
            _ip, _ap, _mp = self.images, self.annotations, self.metadata
            ip = ImageProvider(_ip.df.loc[_ip.df.index.intersection(ids), :], identifier=_ip.identifier)
            ap = AnnotationProvider(_ap.df.loc[_ip.df.index.intersection(ids), :], identifier=_ap.identifier)
            mp = MetadataProvider(_mp.df.loc[_mp.df.index.intersection(ids), :], identifier=_mp.identifier)

        elif callable(ids_or_func):
            func = ids_or_func
            ip = {}
            ap = {}
            mp = {}
            for image_id in self.index:
                item = self.get_by_id(image_id)
                keep = func(item)
                if keep:
                    ip[image_id] = item.image
                    ap[image_id] = item.annotations
                    mp[image_id] = item.metadata

        else:
            raise TypeError(f"requires sequence of ImageId or a callable of type FilterFunc, got {ids_or_func!r}")

        ds = PadoDataset(urlpath, mode="w")
        ds.ingest_obj(ImageProvider(ip, identifier=self.images.identifier))
        ds.ingest_obj(AnnotationProvider(ap, identifier=self.annotations.identifier))
        ds.ingest_obj(MetadataProvider(mp, identifier=self.metadata.identifier))
        return PadoDataset(ds.urlpath, mode="r")

    def partition(
        self,
        splitter: DatasetSplitter,
        label_func: Optional[Callable[[PadoDataset], Sequence[Any]]] = None,
        group_func: Optional[Callable[[PadoDataset], Sequence[Any]]] = None,
    ) -> List[TrainTestDatasetTuple]:
        """partition a pado dataset into train and test

        Parameters
        ----------
        splitter:
            a DatasetSplitter instance (basically all sklearn.model_selection splitter classes)
        label_func:
            gets called with the pado dataset and has to return a sequence of labels with the
            same length as the dataset.index. (default None)
        group_func:
            gets called with the pado dataset and has to return a sequence of groups with the
            same length as the dataset.index. (default None)

        Notes
        -----
        dependent on the provided splitter instance, label_func and group_func might be ignored.

        """
        if label_func is not None:
            labels = label_func(self)
        else:
            labels = None
        if group_func is not None:
            groups = group_func(self)
        else:
            groups = None
        splits = splitter.split(X=self.index, y=labels, groups=groups)
        image_ids = np.array(self.index)

        output = []
        for train_idxs, test_idxs in splits:
            ds0 = self.filter(image_ids[train_idxs])
            ds1 = self.filter(image_ids[test_idxs])
            output.append(TrainTestDatasetTuple(ds0, ds1))
        return output

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
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
            # invalidate caches
            self._cached_index = None
            self._cached_image_provider = None

        elif isinstance(obj, AnnotationProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for AnnotationProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.annotation.parquet")
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
            # invalidate caches
            self._cached_annotation_provider = None

        elif isinstance(obj, MetadataProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for MetadataProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.metadata.parquet")
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
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

        elif store_type == StoreType.ANNOTATION:
            self.ingest_obj(AnnotationProvider.from_parquet(urlpath), identifier=identifier)

        elif store_type == StoreType.METADATA:
            self.ingest_obj(MetadataProvider.from_parquet(urlpath), identifier=identifier)

        else:
            raise NotImplementedError("todo: implement more files")

    # === internal utility methods ===

    def _get_fspath(self, *parts: Union[str, os.PathLike]) -> str:
        """return a fspath for a resource"""
        return os.fspath(pathlib.PurePath(self._root).joinpath(*parts))

    def _ensure_dir(self, *parts: Union[str, os.PathLike]) -> str:
        """ensure that a folder within the dataset exists"""
        fs, pth = self._fs, self._get_fspath(*parts)
        if not fs.isdir(pth):
            fs.mkdir(pth)
        return pth


# === helpers and utils =======================================================

class PadoItem(NamedTuple):
    """A 'row' of a dataset as returned by PadoDataset.get_by_* methods"""
    id: Optional[ImageId]
    image: Optional[Image]
    annotations: Optional[Annotations]
    metadata: Optional[pd.DataFrame]


class TrainTestDatasetTuple(NamedTuple):
    """train test tuple as returned by PadoDataset.partition method"""
    train: PadoDataset
    test: PadoDataset
