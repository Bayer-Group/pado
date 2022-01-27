from __future__ import annotations

import collections.abc
import os
import pathlib
import sys
import textwrap
import uuid
from collections.abc import Iterable
from collections.abc import Sized
from enum import Enum
from typing import Any
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from typing import get_args
from typing import overload

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import fsspec
import pandas as pd
import shapely.wkt

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

__all__ = [
    "PadoDataset",
    "PadoItem",
    "DescribeFormat",
]


class PadoDataset:
    __version__ = 2

    def __init__(
        self,
        urlpath: UrlpathLike | None,
        mode: IOMode = "r",
        *,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
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
        storage_options:
            a optional dictionary with options passed to fsspec for opening the urlpath

        """
        self._storage_options: dict[str, Any] = storage_options or {}

        if urlpath is None:
            # enable in-memory pado datasets and change mode to enable write
            self._urlpath = f"memory://pado-{uuid.uuid4()}"
            mode = "r+"
        else:
            # check provided urlpath and mode
            urlpath = get_root_dir(urlpath, allow_file="pado.dataset.toml")
            try:
                self._urlpath: str = urlpathlike_to_string(urlpath)
            except TypeError as err:
                raise TypeError(f"incompatible urlpath {urlpath!r}") from err

            # check mode
            if mode not in get_args(IOMode):
                raise ValueError(f"unsupported mode {mode!r}")

            # if the dataset files should be there, check them
            fs = self._fs
            if mode in {"r", "r+"}:
                if not list(fs.glob(self._get_fspath("*.image.parquet"))):
                    raise ValueError(
                        f"error: {self._urlpath} not a valid dataset since it has no image parquet file."
                    )
            else:
                pass  # fixme

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
        fs, _ = urlpathlike_to_fs_and_path(self._urlpath, **self._storage_options)
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
                ImageProvider.from_parquet(fsopen(fs, p, mode="rb"))
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
                AnnotationProvider.from_parquet(fsopen(fs, p, mode="rb"))
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
                MetadataProvider.from_parquet(fsopen(fs, p, mode="rb"))
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
        urlpath: Optional[UrlpathLike] = None,
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

        if isinstance(ids_or_func, Iterable) and isinstance(ids_or_func, Sized):
            ids = pd.Series(ids_or_func).apply(str.__call__)
            _ip, _ap, _mp = self.images, self.annotations, self.metadata
            ip = ImageProvider(
                _ip.df.loc[_ip.df.index.intersection(ids), :], identifier=_ip.identifier
            )
            ap = AnnotationProvider(
                _ap.df.loc[_ap.df.index.intersection(ids), :], identifier=_ap.identifier
            )
            mp = MetadataProvider(
                _mp.df.loc[_mp.df.index.intersection(ids), :], identifier=_mp.identifier
            )

        elif callable(ids_or_func):
            func = ids_or_func
            ip = {}
            ap = {}
            mp = {}
            for image_id in self.index:
                item = self.get_by_id(image_id)
                keep = func(item)
                if not keep:
                    continue
                ip[image_id] = item.image
                if item.annotations is not None:
                    ap[image_id] = item.annotations
                if item.metadata is not None:
                    mp[image_id] = item.metadata

        else:
            raise TypeError(
                f"requires sequence of ImageId or a callable of type FilterFunc, got {ids_or_func!r}"
            )

        if len(ip) == 0:
            raise RuntimeError("didn't match any images")

        ds = PadoDataset(urlpath, mode="w")
        ds.ingest_obj(ImageProvider(ip, identifier=self.images.identifier))

        if len(ap) > 0:
            ds.ingest_obj(
                AnnotationProvider(ap, identifier=self.annotations.identifier)
            )
        if len(mp) > 0:
            ds.ingest_obj(MetadataProvider(mp, identifier=self.metadata.identifier))

        return PadoDataset(ds.urlpath, mode="r")

    def partition(
        self,
        splitter: DatasetSplitter,
        label_func: Optional[Callable[[PadoDataset], Sequence[Any]]] = None,
        group_func: Optional[Callable[[PadoDataset], Sequence[Any]]] = None,
    ) -> List[Split]:
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
        image_ids = pd.Series(self.index).values

        output = []
        for train_idxs, test_idxs in splits:
            ds0 = self.filter(image_ids[train_idxs])
            ds1 = self.filter(image_ids[test_idxs])
            output.append(Split(ds0, ds1))
        return output

    # === data ingestion ===

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

    def ingest_file(
        self, urlpath: UrlpathLike, *, identifier: Optional[str] = None
    ) -> None:
        """ingest a file into the dataset"""
        if self.readonly:
            raise RuntimeError(f"{self!r} opened in readonly mode")
        store_type = get_store_type(urlpath)
        if store_type == StoreType.IMAGE:
            self.ingest_obj(ImageProvider.from_parquet(urlpath), identifier=identifier)

        elif store_type == StoreType.ANNOTATION:
            self.ingest_obj(
                AnnotationProvider.from_parquet(urlpath), identifier=identifier
            )

        elif store_type == StoreType.METADATA:
            self.ingest_obj(
                MetadataProvider.from_parquet(urlpath), identifier=identifier
            )

        else:
            raise NotImplementedError("todo: implement more files")

    # === describe (summarise) dataset ===

    @overload
    def describe(self, output_format: Literal[DescribeFormat.PLAIN_TEXT]) -> str:
        ...

    @overload
    def describe(self, output_format: Literal[DescribeFormat.DICT]) -> dict:
        ...

    def describe(
        self, output_format: DescribeFormat = "plain_text"
    ) -> Union[str, dict]:
        """A 'to string' method for essential PadoDataset information"""
        if output_format not in list(DescribeFormat):
            raise ValueError(f"{output_format!r} is not a valid output format.")

        # convert annotations df
        idf = self.images.df
        adf = self.annotations.df
        adf["area"] = adf["geometry"].apply(lambda x: shapely.wkt.loads(x).area)
        agg_annotations = adf.groupby("classification")["area"].agg(["sum", "count"])
        data = {
            "path": self.urlpath,
            "num_images": len(self.images),
            "mean_mpp_x": idf["mpp_x"].mean(),
            "mean_mpp_y": idf["mpp_y"].mean(),
            "mean_image_width": idf["width"].mean(),
            "mean_image_height": idf["height"].mean(),
            "mean_image_size": idf["size_bytes"].mean(),
            "mean_annotations_per_image": adf.groupby("image_id")["geometry"]
            .count()
            .mean(),
            "metadata_columns": self.metadata.df.columns.to_list(),
            "std_mpp_x": idf["mpp_x"].std(),
            "std_mpp_y": idf["mpp_y"].std(),
            "std_image_width": idf["width"].std(),
            "std_image_height": idf["height"].std(),
            "total_size_images": idf["size_bytes"].sum(),
            "total_num_annotations": sum(
                len(x) for x in list(self.annotations.values())
            ),
            "common_classes": list(
                agg_annotations["count"].sort_values(ascending=False)[:5].items()
            ),
            "common_classes_by_annotation_area": list(
                agg_annotations["sum"].sort_values(ascending=False)[:5].items()
            ),
        }

        if output_format in {DescribeFormat.DICT, DescribeFormat.JSON}:
            return data

        elif output_format == DescribeFormat.PLAIN_TEXT:
            return textwrap.dedent(
                """\
                === SUMMARY ===
                Path to dataset: {path}
                Number of images: {num_images}

                === IMAGES ===
                Image Size Distribution (mean, std):
                    - mpp_x ~ ({mean_mpp_x:.3}, {std_mpp_x:.3})
                    - mpp_y ~ ({mean_mpp_y:.3}, {std_mpp_y:.3})
                    - width ~ ({mean_image_width:.3}, {std_image_width:.3})
                    - height ~ ({mean_image_height:.3}, {std_image_height:.3})
                Image File Size (bytes):
                    - mean image size: {mean_image_size_mb:.3f} MB
                    - total size of all images: {total_size_images_gb:.3f} GB

                === ANNOTATIONS ===
                Total number of annotations: {total_num_annotations}
                Mean annotations per image: {mean_annotations_per_image:.3}
                Five most common classes: {common_classes}
                Classes sorted by total annotation area (top five): {common_classes_by_annotation_area}

                === METADATA ===
                Keys available: {metadata_columns}
            """
            ).format(
                mean_image_size_mb=data["mean_image_size"] / 1e6,
                total_size_images_gb=data["total_size_images"] / 1e9,
                **data,
            )

        else:
            raise NotImplementedError(f'Format "{output_format}" is not allowed.')

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


class Split(NamedTuple):
    """train test tuple as returned by PadoDataset.partition method"""

    train: PadoDataset
    test: PadoDataset


class DescribeFormat(str, Enum):
    """supported formats for PadoDataset.describe"""

    PLAIN_TEXT = "plain_text"
    DICT = "dict"
    JSON = "json"
