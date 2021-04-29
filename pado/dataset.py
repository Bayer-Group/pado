from __future__ import annotations
import datetime
import glob
import os
import pathlib
import re
import warnings
from collections.abc import Hashable
from pathlib import Path
from typing import List, Mapping, Optional, Union

import fsspec
import pandas as pd
import toml

from pado.annotations import Annotation, AnnotationResources
from pado.annotations import get_provider as get_annotation_provider
from pado.annotations import store_provider as store_annotation_provider
from pado.datasource import DataSource
from pado.images import ImageId
from pado.images import Image
from pado.images import ImageProvider
from pado.metadata import (
    PadoColumn,
    PadoInvalid,
    PadoReserved,
)
from pado.utils import cached_property, make_chain, make_priority_chain, FilteredMapping, ChainMap

try:
    from typing import Literal, TypedDict  # novermin
except ImportError:
    from typing_extensions import Literal, TypedDict


def is_pado_dataset(path: Path, load_data=False):
    """check if the given path is a valid pado dataset"""
    path = Path(path)
    if path.is_dir():
        path /= "pado.dataset.toml"
    if path.name == "pado.dataset.toml" and path.is_file():
        if not load_data:
            return True
        with path.open("r") as p:
            return toml.load(p)
    else:
        # we could check more, but let's file this under
        # "dataset integrity verification"
        return {} if load_data else False


def verify_pado_dataset_integrity(path: Union[str, os.PathLike]) -> bool:
    """verify file integrity of a pado dataset"""
    path = Path(path)
    data = is_pado_dataset(path, load_data=True)
    if path.is_dir():
        path /= "pado.dataset.toml"
    if not data:
        raise ValueError("provided Path is not a pado dataset")

    dataset_dir = path.parent
    required_dirs = [dataset_dir / "images", dataset_dir / "metadata"]
    for p in required_dirs:
        if not p.is_dir():
            raise ValueError(f"missing {p} directory")

    identifiers = [ds["identifier"] for ds in data["sources"]]
    for identifier in identifiers:
        if not list(glob.glob(os.fspath(dataset_dir / f"metadata/{identifier}.*"))):
            raise ValueError(f"identifier {identifier} is missing metadata")

    return True


class PadoDataItemDict(TypedDict):
    image: Image
    metadata: pd.DataFrame
    annotations: List[Annotation]


class _PadoDataSourceDict(TypedDict):
    identifier: str
    added: datetime.datetime
    num_images: int


class _PadoInfoDict(TypedDict):
    identifier: str
    created: datetime.datetime
    updated: datetime.datetime
    version: int
    sources: List[_PadoDataSourceDict]


class PadoDataset(DataSource):
    __version__ = 1

    def __init__(
        self,
        urlpath: Union[str, pathlib.Path],
        mode: Literal["r", "r+", "w", "w+", "a", "a+", "x", "x+"] = "r",
        identifier: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """open or create a new PadoDataset

        Parameters
        ----------
        path:
            path to `pado.dataset.toml` file, or its parent directory
        mode:
            'r' --> readonly, error if not there
            'r+' --> read/write, error if not there
            'a' = 'a+' --> read/write, create if not there, append if there
            'w' = 'w+' --> read/write, create if not there, truncate if there
            'x' = 'x+' --> read/write, create if not there, error if there
        identifier:
            an optional string identifier in case you want to use a PadoDataset
            as a DataSource for another PadoDataset
        query:
            a query string in the expr format that pandas.DataFrame.query expects
            which can be used to create a subset of a PadoDataset.
            (The query is used on `PadoDataset().metadata`)

        """
        if isinstance(urlpath, pathlib.Path):
            urlpath = re.sub(r"file:/(?!/)", "file:///", urlpath.as_uri())
        else:
            urlpath = os.fspath(urlpath)

        fs, _token, _paths = fsspec.get_fs_token_paths(urlpath)
        self._fs: fsspec.AbstractFileSystem = fs
        self._urlpath: str = urlpath

        self._config = pathlib.PurePath(_paths[0])
        self._mode = mode

        # guarantee p points to `pado.dataset.toml` file (allow directory)
        if not self._config.suffix:
            self._config /= "pado.dataset.toml"
        elif self._config.suffix != ".toml":
            raise ValueError("dataset file requires '.toml' suffix")

        if not re.match(r"^[rawx][+]?$", mode):
            raise ValueError(f"unsupported mode '{mode}'")

        p = self._config
        _exists = self._fs.isfile(os.fspath(self._config))

        self._readonly = mode == "r"
        self._path: pathlib.PurePath = self._config.parent

        if mode in {"r", "r+"} and not _exists:
            raise FileNotFoundError(f'File {p} must exist in "r(+)" mode.')
        elif mode in {"x", "x+"} and _exists:
            raise FileExistsError(f'File {p} must not exist in "x(+)" mode.')
        elif mode in {"w", "w+"} and _exists:
            # todo: truncate
            # shutil.rmtree(self._path, ignore_errors=True)
            # _exists = False
            raise NotImplementedError("not tested yet...")

        # internal paths
        self._path_images = self._path.joinpath("images")
        self._path_metadata = self._path.joinpath("metadata")
        self._path_annotations = self._path.joinpath("annotations")

        # identifier
        if identifier is None:
            identifier = self._path.name

        if _exists:
            self._info = self._load_dataset_toml()
            # check version
            if self.__version__ < self._info["version"]:
                raise RuntimeError("dataset was created with a newer version of pado")

            if identifier != self._info["identifier"]:
                self._info["identifier"] = identifier
                if not self._readonly:
                    self._store_dataset_toml()

            try:
                verify_pado_dataset_integrity(path)
            except ValueError:
                raise RuntimeError("dataset integrity degraded")
        else:
            # ensure folders exist
            self._fs.mkdir(os.fspath(self._path))
            self._fs.mkdir(os.fspath(self._path_images))
            self._fs.mkdir(os.fspath(self._path_metadata))
            self._fs.mkdir(os.fspath(self._path_annotations))
            # write initial dataset toml
            self._info = self._store_dataset_toml(_info={"identifier": identifier})

        # cached metadata dataframe and image_provider
        self._metadata_df = None
        self._metadata_col_map = None
        self._image_provider = None
        self._annotations_provider = None

        # keep query string
        self._metadata_query_str = query

        # image_id column cache
        self._metadata_df_image_id_col = None

    def query(self, query_str: str) -> PadoDataset:
        """ Simplest implementation of querying from an existing PadoDataset"""
        # TODO: improve this such that operations like .query().query() make sense
        return PadoDataset(urlpath=self._urlpath,
                           mode=self._mode,
                           query=query_str)

    @property
    def path(self):
        """root folder of pado dataset"""
        return self._path

    @property
    def urlpath(self):
        """the fsspec urlpath for this dataset"""
        return self._urlpath

    @property
    def filesystem(self) -> fsspec.AbstractFileSystem:
        """the fsspec filesystem for this dataset"""
        return self._fs

    @property
    def identifier(self):
        return self._info["identifier"]

    @property
    def images(self) -> ImageProvider:
        """a sequence-like interface to all images in the dataset"""
        if self._image_provider is None:
            self._image_provider = make_chain([
                ImageProvider.from_parquet(self._path_images)
                for p in glob.glob(os.fspath(self._path_images / "*")) if os.path.isdir(p)
            ])

            if self._metadata_query_str is not None:
                # Marco's fix: unique_image_ids are strings, while valid_keys need to be ImageId
                # unique_image_ids = sorted(self.metadata[PadoColumn.IMAGE].unique())

                # This function is already used twice in the class. Shall we define it outside?
                def _convert_to_image_id(x):
                    try:
                        return ImageId.from_str(x)
                    except ValueError:
                        return ImageId(*x.split("__"))

                unique_image_ids = sorted(self.metadata[PadoColumn.IMAGE].apply(_convert_to_image_id).unique())
                self._image_provider = FilteredMapping(self._image_provider, valid_keys=unique_image_ids)

        return self._image_provider

    def _reassociate_images(self, *paths: Union[str, Path]):
        if self._image_provider is None:
            _ = self.images
        if isinstance(self._image_provider, FilteredMapping):
            raise NotImplementedError("not supported for filtered datasets")
        elif isinstance(self._image_provider, ChainMap):
            for ip in self._image_provider.maps:
                for path in paths:
                    ip.reassociate_resources(path)
                ip.save()
        elif isinstance(self._image_provider, ImageProvider):
            for path in paths:
                self._image_provider.reassociate_resources(path)
            self._image_provider.save()
        else:
            raise NotImplementedError(f"unexpected image provider type '{self._image_provider}'")

    @property
    def metadata(self) -> pd.DataFrame:
        """a pandas DataFrame providing all metadata stored in the dataset"""
        _ext = ".parquet"

        if self._metadata_df is None:
            md_dir = self._path_metadata
            dfs, keys = [], []
            for metadata_path in self._fs.glob(f"{self._path_metadata}/*{_ext}"):
                with self._fs.open(metadata_path) as f:
                    dfs.append(pd.read_parquet(f))
                    keys.append(os.path.basename(metadata_path)[: -len(_ext)])
            # build the combined df and allow differentiating data sources
            df = pd.concat(dfs, keys=keys)
            # this implicitly assumes that "level_0" is reserved
            df = (
                df.reset_index(level=0)
                .rename(
                    columns={
                        PadoInvalid.RESERVED_COL_INDEX: PadoReserved.DATA_SOURCE_ID
                    }
                )
                .reset_index(drop=True)
            )
            self._metadata_df = df

            if self._metadata_query_str is not None:
                self._metadata_df.query(expr=self._metadata_query_str, inplace=True)

        return self._metadata_df

    @property
    def annotations(self) -> Mapping[ImageId, AnnotationResources]:
        """a mapping-like interface for all annotations per image"""
        if self._annotations_provider is None:
            self._annotations_provider = make_chain([
                get_annotation_provider(p)
                for p in glob.glob(os.fspath(self._path_annotations / "*")) if os.path.isdir(p)
            ])
        return self._annotations_provider

    def __iter__(self):
        yield from self.images

    def __getitem__(self, item: ImageId) -> PadoDataItemDict:
        image = self.images[item]
        if isinstance(image, Image):
            warnings.warn(
                "you're requesting data from a dataset that contains remote image resources"
            )  # pragma: no cover
        _df = self.metadata

        if self._metadata_df_image_id_col is None:
            def _convert_to_image_id(x):
                try:
                    return ImageId.from_str(x)
                except ValueError:
                    return ImageId(*x.split("__"))  # legacy support

            self._metadata_df_image_id_col = _df[PadoColumn.IMAGE].apply(_convert_to_image_id)

        metadata = _df[self._metadata_df_image_id_col == item]
        if metadata.size == 0:
            warnings.warn(f"no metadata for {item!r}")

        try:
            annotation_dict = self.annotations[item].copy()
        except KeyError:
            annotations = []
        else:
            annotations = annotation_dict.pop("annotations", [])
        # TODO: annotation_dict should be included in metadata_dict
        return PadoDataItemDict(
            image=image, metadata=metadata, annotations=annotations,
        )

    def __len__(self):
        return len(self.images)

    def _fspath(self, *parts):
        return os.fspath(pathlib.PurePath().joinpath(*parts))

    def _fsopen(self, parts, mode="rb"):
        if isinstance(parts, (str, Path)):
            parts = (parts,)
        return self._fs.open(self._fspath(*parts), mode=mode)

    def add_source(self, source: DataSource, copy_images: bool = True):
        if self._readonly:
            raise RuntimeError("Can't add sources to readonly dataset")

        # store metadata and images
        with source:
            self._store_metadata(source)
            self._store_image_provider(source, copy_images)
            self._store_annotation_provider(source)
            self._store_dataset_toml(add_source=source)

    def _store_metadata(self, source: DataSource):
        """store the metadata in the dataset"""
        identifier = source.identifier
        metadata_path = self._fsopen((self._path_metadata, f"{identifier}.parquet"), mode="wb")
        source.metadata.to_parquet(metadata_path, compression="gzip")
        # clear cache
        self._metadata_df = self._metadata_col_map = None

    def _store_image_provider(
        self,
        source: DataSource,
        copy_images: bool = True,
    ):
        """store the image provider to the dataset"""
        identifier = source.identifier

        ip = ImageProvider(source.images)
        if copy_images:
            # ingest images
            for image_id, image in ip.items():
                old_pth = os.fspath(image.fspath)
                new_pth = self._fspath(self._path, self._path_images, image_id.site, image_id.to_path())
                self._fs.mkdirs(os.path.dirname(new_pth), exist_ok=True)
                self._fs.copy(old_pth, new_pth)

        ip.to_parquet(
            fspath=self._fsopen((self._path_images, f"{identifier}.parquet"), mode="wb")
        )
        # clear cache
        self._image_provider = None
        self._metadata_df_image_id_col = None

    def _store_annotation_provider(self, source: DataSource):
        """store the annotation provider to the dataset"""
        path = self._path_annotations / source.identifier
        store_annotation_provider(path, source.annotations)
        # clear cache
        self._annotations_provider = None

    def _store_dataset_toml(
        self, add_source: Optional[DataSource] = None, *, _info=None
    ) -> _PadoInfoDict:
        if _info is None:
            _info = self._info

        info_dict = _PadoInfoDict(
            identifier=_info["identifier"],
            created=_info.get("created", datetime.datetime.now()),
            updated=datetime.datetime.now(),
            version=self.__version__,
            sources=_info.get("sources", []),
        )
        if add_source:
            source_dict = _PadoDataSourceDict(
                identifier=add_source.identifier,
                added=datetime.datetime.now(),
                num_images=len(add_source.images),
            )
            info_dict["sources"].append(source_dict)

        with self._fs.open(os.fspath(self._config), mode="w") as config:
            toml.dump(info_dict, config)
        return info_dict

    def _load_dataset_toml(self) -> _PadoInfoDict:
        with self._config.open("r") as config:
            dataset_config = toml.load(config)
        return dataset_config


class PadoDatasetChain:
    """Chain multiple pado datasets"""

    def __init__(self, *datasets: PadoDataset):
        self._datasets = list(datasets)
        self._metadata_col_map = {}

    @property
    def path(self):
        """root folder of first pado dataset"""
        return self._datasets[0].path

    @staticmethod
    def _first_exists_fallback_last_(resources, default):
        r = default
        for r in resources:
            pth: Optional[Path] = r.local_path
            if pth and pth.is_file():
                return r
        else:
            return r

    @cached_property
    def images(self) -> ImageProvider:
        """images in the pado dataset"""
        return make_priority_chain(
            (ds.images for ds in self._datasets),
            priority_func=self._first_exists_fallback_last_
        )

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """combined dataframe"""
        df = pd.concat([ds.metadata for ds in self._datasets])

        # some columns stored in dataframes might not be hashable...
        hashable_columns = []
        for col in df:
            series = df[col]
            column_types = series.apply(type).unique()
            if all(issubclass(t, Hashable) for t in column_types):
                hashable_columns.append(col)

        df.drop_duplicates(subset=hashable_columns, keep="first", inplace=True)
        return df

    @cached_property
    def annotations(self) -> Mapping[str, AnnotationResources]:
        """chaining annotations together"""
        return make_chain([ds.annotations for ds in self._datasets])

    __iter__ = PadoDataset.__iter__
    __getitem__ = PadoDataset.__getitem__
    __len__ = PadoDataset.__len__
