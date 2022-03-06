from __future__ import annotations

import os
import pathlib
import sys
import uuid
from collections.abc import Iterable
from collections.abc import Sized
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

from pado._compat import cached_property
from pado._repr import DescribeFormat
from pado._repr import describe_format_plain_text
from pado._repr import number
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
from pado.predictions.providers import ImagePredictionProvider
from pado.predictions.proxy import PredictionProxy
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
                if not (
                    fs.isdir(self._root)  # raises if not there or reachable
                    and any(fs.glob(self._get_fspath("*.image.parquet")))
                ):
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

    @property
    def urlpath(self) -> str:
        """the urlpath pointing to the PadoDataset"""
        return self._urlpath

    @property
    def _fs(self) -> fsspec.AbstractFileSystem:
        fs, _ = urlpathlike_to_fs_and_path(
            self._urlpath, storage_options=self._storage_options
        )
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
        so = ""
        if self._storage_options:
            so = f", storage_options={self._storage_options!r}"
        return f"{type(self).__name__}({self.urlpath!r}, mode={self._mode!r}{so})"

    # === data properties ===

    @cached_property
    def index(self) -> Sequence[ImageId]:
        """sequence of image_ids in the dataset"""
        image_ids = self.images.keys()
        if isinstance(image_ids, Sequence):
            index = image_ids
        else:
            index = tuple(image_ids)
        return index

    @cached_property
    def images(self) -> ImageProvider:
        """mapping image_ids to images in the dataset"""
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

        return image_provider

    @cached_property
    def annotations(self) -> AnnotationProvider:
        """mapping image_ids to annotations in the dataset"""
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

        return annotation_provider

    @cached_property
    def metadata(self) -> MetadataProvider:
        """mapping image_ids to metadata in the dataset"""
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

        return metadata_provider

    @cached_property
    def predictions(self) -> PredictionProxy:
        return PredictionProxy(self)

    # === access ===

    def get_by_id(self, image_id: ImageId) -> PadoItem:
        try:
            return PadoItem(
                image_id,
                self.images[image_id],
                self.annotations.get(image_id),
                self.metadata.get(image_id),
            )
        except KeyError:
            raise KeyError(f'{image_id} does not match any images in this dataset.')

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
        elif isinstance(obj, dict):
            raise NotImplementedError("todo: guess provider type")

        elif isinstance(obj, ImageProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for ImageProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.image.parquet")
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
            # invalidate caches
            self._clear_caches("images")

        elif isinstance(obj, AnnotationProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for AnnotationProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.annotation.parquet")
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
            # invalidate caches
            self._clear_caches("annotations")

        elif isinstance(obj, MetadataProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError("need to provide an identifier for MetadataProvider")
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.metadata.parquet")
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
            # invalidate caches
            self._clear_caches("metadata")

        elif isinstance(obj, ImagePredictionProvider):
            if identifier is None and obj.identifier is None:
                raise ValueError(
                    "need to provide an identifier for ImagePredictionProvider"
                )
            identifier = identifier or obj.identifier
            pth = self._get_fspath(f"{identifier}.image_predictions.parquet")
            obj.to_parquet(fsopen(self._fs, pth, mode="xb"))
            # invalidate caches
            self._clear_caches("predictions")

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
    def describe(self) -> dict:
        ...

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
            "num_mpps": [
                {"mpp": k, "num": v}
                for k, v in idf[["mpp_x", "mpp_y"]].value_counts().items()
            ],
            "avg_image_width": number(idf["width"], agg="avg", unit="px"),
            "avg_image_height": number(idf["height"], agg="avg", unit="px"),
            "avg_image_size": number(idf["size_bytes"], agg="avg", unit="b"),
            "avg_annotations_per_image": number(
                adf.groupby("image_id")["geometry"].count(), agg="avg"
            ),
            "metadata_columns": self.metadata.df.columns.to_list(),
            "total_size_images": number(idf["size_bytes"], agg="sum", unit="b"),
            "total_num_annotations": sum(
                len(x) for x in list(self.annotations.values())
            ),
            "common_classes_count": dict(
                agg_annotations["count"].sort_values(ascending=False)[:5].items()
            ),
            "common_classes_area": {
                k: number(v, cast=float, unit="px")
                for k, v in agg_annotations["sum"]
                .sort_values(ascending=False)[:5]
                .items()
            },
        }

        if output_format in {DescribeFormat.DICT, DescribeFormat.JSON}:
            return data

        elif output_format == DescribeFormat.PLAIN_TEXT:
            return describe_format_plain_text(data)

        else:
            raise NotImplementedError(f'Format "{output_format}" is not allowed.')

    # === internal utility methods ===

    def _clear_caches(
        self, *caches: Literal["images", "metadata", "annotations", "predictions"]
    ) -> None:
        """clear each requested cached_property"""
        valid_caches = ("images", "metadata", "annotations", "predictions")
        if not caches:
            caches = valid_caches
        elif not set(caches).issubset(valid_caches):
            raise ValueError(
                f"unsupported cache: {set(caches).difference(valid_caches)}"
            )
        caches = list(caches)
        if "images" in caches:
            caches.insert(caches.index("images") + 1, "index")
        for cache in reversed(caches):
            try:
                del self.__dict__[cache]
            except KeyError:
                pass

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
