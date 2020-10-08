from abc import ABC, abstractmethod
from typing import Mapping

import pandas as pd

from pado.annotations import AnnotationResources
from pado.images import ImageResourcesProvider


class DataSource(ABC):
    """DataSource base class

    All data sources should go through this abstraction to
    allow channelling them into the same pado.DataSet format.

    """

    identifier: str

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        raise NotImplementedError("implement in subclass")

    @property
    @abstractmethod
    def images(self) -> ImageResourcesProvider:
        raise NotImplementedError("implement in subclass")

    @property
    @abstractmethod
    def annotations(self) -> Mapping[str, AnnotationResources]:
        raise NotImplementedError("implement in subclass")

    def acquire(self, raise_if_missing: bool = False):
        pass

    def release(self):
        pass

    def __enter__(self):
        # prevent implicitly doing potentially expensive things
        self.acquire(raise_if_missing=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def verify_datasource(ds: DataSource, acquire=False):
    from pado.metadata import (
        PadoColumn,
        build_column_map,
        structurize_metadata,
        verify_columns,
    )

    if acquire:
        ds.acquire()
    with ds:
        image_ids = set(img.id_str for img in ds.images)
        # no images with duplicated ids
        if len(ds.images) != len(image_ids):
            raise ValueError("contains images with duplicated ids")
        # metadata columns are valid
        df = ds.metadata
        verify_columns(df.columns, raise_if_invalid=True)
        # every image_id is present in metadata
        if image_ids != set(df[PadoColumn.IMAGE]):
            raise ValueError(
                "metadata IMAGE column must have 1 or more rows for each image"
            )
        # verify the metadata structure per image
        col_map = build_column_map(df.columns)
        for image_id in image_ids:
            md = df[df[PadoColumn.IMAGE] == image_id]
            try:
                structurize_metadata(md, PadoColumn.IMAGE, col_map)
            except Exception:
                raise ValueError("metadata structure has problems")
    return ds
