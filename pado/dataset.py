from __future__ import annotations

import datetime
import os
import pathlib
import re
from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import pandas as pd
import toml

from pado.resource import (
    DataSource,
    ImageResourceCopier,
    ImageResourcesProvider,
    MergedImageResourcesProvider,
    SerializableImageResourcesProvider,
)
from pado.structure import (
    PadoColumn,
    PadoInvalid,
    PadoReserved,
    build_column_map,
    verify_columns,
)

PathOrStr = Union[str, os.PathLike]


def is_pado_dataset(path: Path, load_data=False):
    """check if the given path is a valid pado dataset"""
    path = Path(path)
    if path.is_dir():
        path /= "pado.dataset.toml"
    if path.name == "pado.dataset.toml" and path.is_file():
        if not load_data:
            return True
        with path.open("rb") as p:
            return toml.load(p)
    else:
        # we could check more, but let's file this under
        # "dataset integrity verification"
        return {} if load_data else False


def verify_pado_dataset_integrity(path: PathOrStr) -> bool:
    """verify file integrity of a pado dataset"""
    path = Path(path)
    data = is_pado_dataset(path, load_data=True)
    if not data:
        raise ValueError("provided Path is not a pado dataset")

    dataset_dir = path.parent
    required_dirs = map(dataset_dir.__div__, ["images", "metadata"])
    if not all(p.is_dir() for p in required_dirs):
        return False

    identifiers = [ds["identifier"] for ds in data["sources"]]
    for identifier in identifiers:
        if not path.glob(f"metadata/{identifier}.*"):
            return False

    return True


@pd.api.extensions.register_dataframe_accessor("pado")
class PadoAccessor:
    """provide pado specific operations on the dataframe"""

    c = PadoColumn
    """provide shorthand for standardized columns"""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._df = pandas_obj
        self._cm = build_column_map(pandas_obj.columns)

    @staticmethod
    def _validate(obj: pd.DataFrame):
        """validate the provided dataframe"""
        # check required columns
        if not set(PadoColumn).issubset(obj.columns):
            missing = set(PadoColumn) - set(obj.columns)
            mc = ", ".join(map(str.__repr__, sorted(missing)))
            raise AttributeError(f"missing columns: {mc}")
        # check if columns are compliant
        try:
            verify_columns(columns=obj.columns, raise_if_invalid=True)
        except ValueError as err:
            raise AttributeError(str(err))

    def _subset(self, column: PadoColumn) -> pd.DataFrame:
        """return the dataframe subset belonging to a PadoColumn"""
        return self._df.loc[:, self._cm[column]].drop_duplicates()

    class _SubsetDescriptor:
        """descriptor for accessing the dataframe subsets"""

        def __init__(self, pado_column: PadoColumn):
            self._col = pado_column

        def __get__(self, instance, owner):
            if instance is None:
                return self
            # noinspection PyProtectedMember
            return instance._subset(self._col)

    # the dataframe accessors
    studies = _SubsetDescriptor(c.STUDY)
    experiments = _SubsetDescriptor(c.EXPERIMENT)
    groups = _SubsetDescriptor(c.GROUP)
    animals = _SubsetDescriptor(c.ANIMAL)
    compounds = _SubsetDescriptor(c.COMPOUND)
    organs = _SubsetDescriptor(c.ORGAN)
    slides = _SubsetDescriptor(c.SLIDE)
    images = _SubsetDescriptor(c.IMAGE)
    findings = _SubsetDescriptor(c.FINDING)


DatasetIOMode = Union[
    Literal["r"],
    Literal["r+"],
    Literal["w"],
    Literal["w+"],
    Literal["a"],
    Literal["a+"],
    Literal["x"],
    Literal["x+"],
]


class PadoDataSourceDict(TypedDict):
    identifier: str
    added: datetime.datetime
    num_images: int


class PadoInfoDict(TypedDict):
    identifier: str
    created: datetime.datetime
    updated: datetime.datetime
    version: int
    sources: List[PadoDataSourceDict]


class PadoDataset(DataSource):
    __version__ = 1

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        mode: DatasetIOMode = "r",
        identifier: Optional[str] = None,
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

        """
        self._config = pathlib.Path(path)
        self._mode = str(mode)

        # guarantee p points to `pado.dataset.toml` file (allow directory)
        if not self._config.suffix:
            self._config /= "pado.dataset.toml"
        elif self._config.suffix != ".toml":
            raise ValueError("dataset file requires '.toml' suffix")

        if not re.match(r"^[rawx][+]?$", mode):
            raise ValueError(f"unsupported mode '{mode}'")

        p = self._config.expanduser().absolute()
        _exists = p.is_file()

        self._readonly = mode == "r"
        self._path = self._config.parent

        if mode in {"r", "r+"} and not _exists:
            raise FileNotFoundError(p)
        elif mode in {"x", "x+"} and _exists:
            raise FileExistsError(p)
        elif mode in {"w", "w+"} and _exists:
            # todo: truncate
            # shutil.rmtree(self._path, ignore_errors=True)
            _exists = False
            raise NotImplementedError("not tested yet...")

        # internal paths
        self._path_images = self._path / "images"
        self._path_metadata = self._path / "metadata"

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

            if not verify_pado_dataset_integrity(path):
                raise RuntimeError("dataset integrity degraded")
        else:
            # ensure folders exist
            self._path.mkdir(exist_ok=True)
            self._path_images.mkdir(exist_ok=True)
            self._path_metadata.mkdir(exist_ok=True)
            # write initial dataset toml
            self._info = self._store_dataset_toml(_info={"identifier": identifier})

        # cached metadata dataframe and image_provider
        self._metadata_df = None
        self._image_provider = None

    @property
    def path(self):
        """root folder of pado dataset"""
        return self._path

    @property
    def identifier(self):
        return self._info["identifier"]

    @property
    def metadata(self) -> pd.DataFrame:
        _ext = ".parquet.gzip"

        if self._metadata_df is None:
            md_dir = self._path_metadata
            dfs, keys = [], []
            for metadata_file in md_dir.glob(f"*{_ext}"):
                dfs.append(pd.read_parquet(metadata_file))
                keys.append(metadata_file.name[: -len(_ext)])
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
        return self._metadata_df

    @property
    def images(self) -> ImageResourcesProvider:
        if self._image_provider is None:
            providers = []
            for p in self._path_images.glob("*"):
                if not p.is_dir():
                    continue
                providers.append(
                    SerializableImageResourcesProvider(p.name, self._path_images)
                )
            self._image_provider = MergedImageResourcesProvider(providers)
        return self._image_provider

    def add_source(self, source: DataSource, copy_images: bool = True):
        if self._readonly:
            raise RuntimeError("can't add sources to readonly dataset")

        # store metadata and images
        with source:
            self._store_metadata(source)
            self._store_image_provider(source, copy_images)
            self._store_dataset_toml(add_source=source)

    def _store_metadata(self, source):
        """store the metadata in the dataset"""
        identifier = source.identifier
        metadata_path = self._path_metadata / f"{identifier}.parquet.gzip"
        if metadata_path.is_file():
            # todo: allow extending
            raise ValueError("source already exists")

        source.metadata.to_parquet(metadata_path, compression="gzip")

    def _store_image_provider(self, source, copy_images=True, copier=None):
        """store the image provider to the dataset"""
        identifier = source.identifier

        ip = SerializableImageResourcesProvider.from_provider(
            identifier, self._path_images, source.images
        )
        if copy_images:
            if copier is None:
                copier = ImageResourceCopier(identifier, self._path_images)
            copier(ip)

    def _store_dataset_toml(
        self, add_source: Optional[DataSource] = None, *, _info=None
    ) -> PadoInfoDict:
        if _info is None:
            _info = self._info

        info_dict = PadoInfoDict(
            identifier=_info["identifier"],
            created=_info.get("created", datetime.datetime.now()),
            updated=datetime.datetime.now(),
            version=self.__version__,
            sources=_info.get("sources", []),
        )
        if add_source:
            source_dict = PadoDataSourceDict(
                identifier=add_source.identifier,
                added=datetime.datetime.now(),
                num_images=len(add_source.images),
            )
            info_dict["sources"].append(source_dict)

        with self._config.open("w") as config:
            toml.dump(info_dict, config)
        return info_dict

    def _load_dataset_toml(self) -> PadoInfoDict:
        with self._config.open("r") as config:
            dataset_config = toml.load(config)
        return dataset_config
