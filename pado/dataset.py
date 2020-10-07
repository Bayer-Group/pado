import datetime
import os
import pathlib
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Union

import pandas as pd
import toml

from pado.annotations import AnnotationResources
from pado.annotations import get_provider as get_annotation_provider
from pado.annotations import merge_providers as merge_annotation_providers
from pado.annotations import store_provider as store_annotation_provider
from pado.datasource import DataSource
from pado.images import (
    ImageResourceCopier,
    ImageResourcesProvider,
    MergedImageResourcesProvider,
    RemoteImageResource,
    SerializableImageResourcesProvider,
)
from pado.metadata import (
    PadoColumn,
    PadoInvalid,
    PadoReserved,
    build_column_map,
    structurize_metadata,
)

try:
    from typing import Literal, TypedDict  # novermin
except ImportError:
    from typing_extensions import Literal, TypedDict


PathOrStr = Union[str, os.PathLike]


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


def verify_pado_dataset_integrity(path: PathOrStr) -> bool:
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
        if not list(dataset_dir.glob(f"metadata/{identifier}.*")):
            raise ValueError(f"identifier {identifier} is missing metadata")

    return True


DatasetIOMode = Literal["r", "r+", "w", "w+", "a", "a+", "x", "x+"]


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
            # _exists = False
            raise NotImplementedError("not tested yet...")

        # internal paths
        self._path_images = self._path / "images"
        self._path_metadata = self._path / "metadata"
        self._path_annotations = self._path / "annotations"

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
            self._path.mkdir(exist_ok=True)
            self._path_images.mkdir(exist_ok=True)
            self._path_metadata.mkdir(exist_ok=True)
            self._path_annotations.mkdir(exist_ok=True)
            # write initial dataset toml
            self._info = self._store_dataset_toml(_info={"identifier": identifier})

        # cached metadata dataframe and image_provider
        self._metadata_df = None
        self._metadata_col_map = None
        self._image_provider = None
        self._annotations_provider = None

    @property
    def path(self):
        """root folder of pado dataset"""
        return self._path

    @property
    def identifier(self):
        return self._info["identifier"]

    @property
    def images(self) -> ImageResourcesProvider:
        """a sequence-like interface to all images in the dataset"""
        if self._image_provider is None:
            providers = []
            for p in filter(os.path.isdir, self._path_images.glob("*")):
                providers.append(
                    SerializableImageResourcesProvider(p.name, self._path_images)
                )
            self._image_provider = MergedImageResourcesProvider(providers)
        return self._image_provider

    @property
    def metadata(self) -> pd.DataFrame:
        """a pandas DataFrame providing all metadata stored in the dataset"""
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
            self._metadata_col_map = build_column_map(df.columns)
        return self._metadata_df

    @property
    def annotations(self) -> Mapping[str, AnnotationResources]:
        """a mapping-like interface for all annotations per image"""
        if self._annotations_provider is None:
            providers = []
            for p in filter(os.path.isdir, self._path_annotations.glob("*")):
                providers.append(get_annotation_provider(p))
            self._annotations_provider = merge_annotation_providers(providers)
        return self._annotations_provider

    def __getitem__(self, item: int) -> Dict:
        image = self.images[item]
        if isinstance(image, RemoteImageResource):
            warnings.warn(
                "you're requesting data from a dataset that contains remote image resources"
            )  # pragma: no cover
        _df = self.metadata
        metadata = _df[_df[PadoColumn.IMAGE] == image.id_str]
        # this is where a relational database would be great...
        metadata_dict = structurize_metadata(
            metadata, PadoColumn.IMAGE, self._metadata_col_map
        )
        annotation_dict = self.annotations.get(image.id_str, {}).copy()
        annotations = annotation_dict.pop("annotations", [])
        # TODO: annotation_dict should be included in metadata_dict
        return {
            "image": image,
            "metadata": metadata_dict,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.images)

    def add_source(self, source: DataSource, copy_images: bool = True):
        if self._readonly:
            raise RuntimeError("can't add sources to readonly dataset")

        # store metadata and images
        with source:
            self._store_metadata(source)
            self._store_image_provider(source, copy_images)
            self._store_annotation_provider(source)
            self._store_dataset_toml(add_source=source)

    def _store_metadata(self, source: DataSource):
        """store the metadata in the dataset"""
        identifier = source.identifier
        metadata_path = self._path_metadata / f"{identifier}.parquet.gzip"
        if metadata_path.is_file():
            # todo: allow extending
            raise ValueError("source already exists")

        source.metadata.to_parquet(metadata_path, compression="gzip")

    def _store_image_provider(
        self,
        source: DataSource,
        copy_images: bool = True,
        copier: Optional[Callable[[ImageResourcesProvider], None]] = None,
    ):
        """store the image provider to the dataset"""
        identifier = source.identifier

        ip = SerializableImageResourcesProvider.from_provider(
            identifier, self._path_images, source.images
        )
        if copy_images:
            if copier is None:
                copier = ImageResourceCopier(identifier, self._path_images)
            copier(ip)

    def _store_annotation_provider(self, source: DataSource):
        """store the annotation provider to the dataset"""
        path = self._path_annotations / source.identifier
        store_annotation_provider(path, source.annotations)

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
