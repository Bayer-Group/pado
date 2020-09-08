from __future__ import annotations

import os
import pathlib
import re
import shutil
from pathlib import Path
from typing import Iterable, Literal, Union

import pandas as pd

from pado.datasource import DataSource, ImageResource
from pado.structure import (
    PadoColumn,
    PadoInvalid,
    PadoReserved,
    build_column_map,
    verify_columns,
)

PathOrStr = Union[str, os.PathLike]


def is_pado_dataset(path: PathOrStr) -> bool:
    """check if the given path is a valid pado dataset"""
    path = Path(path)
    if path.is_dir():
        path /= "pado.dataset.toml"
    if path.name == "pado.dataset.toml" and path.is_file():
        return True
    else:
        # we could check more, but let's file this under
        # "dataset integrity verification"
        return False


def verify_pado_dataset_integrity(path: PathOrStr) -> bool:
    """verify file integrity of a pado dataset"""
    # fixme: skeleton implementation
    # todo:
    #   - verify file hashes
    #   - add specific file or store ?in pado.dataset.toml?
    #
    path = Path(path)
    if not is_pado_dataset(path):
        raise ValueError("provided Path is not a pado dataset")

    dataset_dir = path.parent
    for file in dataset_dir.glob("**/*"):
        # todo: check integrity
        pass

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


class PadoDataset(DataSource):
    def __init__(self, path: Union[str, pathlib.Path], mode: DatasetIOMode = "r"):
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

        if _exists:
            if not verify_pado_dataset_integrity(path):
                raise RuntimeError("dataset integrity degraded")
        else:
            pass

        # cached metadata dataframe
        self._metadata_df = None
        self._path_images = self._path / "images"
        self._path_metadata = self._path / "metadata"

        # ensure folders exist
        self._path.mkdir(exist_ok=True)
        self._path_images.mkdir(exist_ok=True)
        self._path_metadata.mkdir(exist_ok=True)

    @property
    def path(self):
        """root folder of pado dataset"""
        return self._path

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

    def images(self) -> Iterable[ImageResource]:
        raise NotImplementedError("todo: implement iteration over multiple sources")

    def add_source(self, source: DataSource, copy_images: bool = True):
        if self._readonly:
            raise RuntimeError("can't add sources to readonly dataset")
        identifier = source.identifier

        metadata_path = self._path_metadata / f"{identifier}.parquet.gzip"
        if metadata_path.is_file():
            # todo: allow extending
            raise ValueError("source already exists")

        # store metadata
        source.metadata.to_parquet(metadata_path, compression="gzip")

        base = self._path_images / identifier
        base.mkdir(exist_ok=True)
        for image in source.images():
            dst = base / Path(*image.id)
            # note: maybe replace with urlretrieve to allow downloading
            if copy_images:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(image.path, dst)
            else:
                raise NotImplementedError("todo: allow keeping references?")
