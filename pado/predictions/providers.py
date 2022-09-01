from __future__ import annotations

import enum
import json
import uuid
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import MutableMapping

import pandas as pd

from pado._repr import mapping_repr
from pado.collections import GroupedProviderMixin
from pado.collections import PadoMutableSequence
from pado.collections import PadoMutableSequenceMapping
from pado.collections import ProviderStoreMixin
from pado.collections import SerializableProviderMixin
from pado.images.ids import ImageId
from pado.images.image import Image
from pado.images.utils import Bounds
from pado.io.store import Store
from pado.io.store import StoreType

# === ImagePredictions ========================================================


class ImagePredictionType(str, enum.Enum):
    HEATMAP = "heatmap"


class ImagePrediction:
    def __init__(
        self,
        image_id: ImageId,
        prediction_type: ImagePredictionType,
        bounds: Bounds,
        extra_metadata: dict[str, Any],
        image: Image,
    ):
        self.image_id: ImageId = image_id
        self.prediction_type: ImagePredictionType = prediction_type
        self.bounds: Bounds = bounds
        self.extra_metadata: dict[str, Any] = extra_metadata
        self.image: Image = image

    __fields__: tuple[str, ...] = tuple(
        [
            "image_id",
            "prediction_type",
            "bounds",
            "extra_metadata",
            *Image.__fields__,
        ]
    )

    @classmethod
    def from_obj(cls, obj: Any) -> ImagePrediction:
        if not isinstance(obj, pd.Series):
            raise NotImplementedError("todo")
        else:
            return cls(
                image_id=ImageId.from_str(obj.pop("image_id")),
                prediction_type=ImagePredictionType(obj.pop("prediction_type")),
                bounds=Bounds.from_record(json.loads(obj.pop("bounds"))),
                extra_metadata=json.loads(obj.pop("extra_metadata")),
                image=Image.from_obj(obj),
            )

    def to_record(self, image_id: ImageId | None = None) -> dict[str, Any]:

        if self.image_id is not None and image_id is not None:
            if self.image_id != image_id:
                raise ValueError(
                    f"ImagePrediction has different image_id: has {self.image_id} requested {image_id}"
                )

        return {
            "image_id": self.image_id.to_str(),
            "prediction_type": self.prediction_type.value,
            "bounds": json.dumps(self.bounds.as_record()),
            "extra_metadata": json.dumps(self.extra_metadata),
            **self.image.to_record(
                urlpath_ignore_options=("profile",)  # fixme: expose
            ),
        }


class ImagePredictions(PadoMutableSequence[ImagePrediction]):
    __item_class__ = ImagePrediction


class ImagePredictionsProviderStore(ProviderStoreMixin, Store):
    """stores the image predictions provider in a single file with metadata"""

    METADATA_KEY_PROVIDER_VERSION = "image_predictions_provider_version"
    PROVIDER_VERSION = 1

    def __init__(
        self, version: int = 1, store_type: StoreType = StoreType.IMAGE_PREDICTIONS
    ):
        if store_type != StoreType.IMAGE_PREDICTIONS:
            raise ValueError("changing store_type in subclasses unsupported")
        super().__init__(version=version, store_type=store_type)


class ImagePredictionProvider(
    PadoMutableSequenceMapping[ImagePredictions], SerializableProviderMixin
):
    __store_class__ = ImagePredictionsProviderStore
    __value_class__ = ImagePredictions


class GroupedImagePredictionProvider(GroupedProviderMixin, ImagePredictionProvider):
    __provider_class__ = ImagePredictionProvider


# === MetadataPredictions =====================================================


# intermediate compromise
class MetadataPrediction:
    __fields__ = (
        "image_id",
        "model_extra_json",  # currently, encoding some sort of model id
        "row_json",  # for now to ease migration later
    )


class MetadataPredictionsProviderStore(ProviderStoreMixin, Store):
    """stores the metadata predictions provider in a single file with metadata"""

    METADATA_KEY_PROVIDER_VERSION = "metadata_predictions_provider_version"
    PROVIDER_VERSION = 1

    def __init__(
        self, version: int = 1, store_type: StoreType = StoreType.METADATA_PREDICTIONS
    ):
        if store_type != StoreType.ANNOTATION:
            raise ValueError("changing store_type in subclasses unsupported")
        super().__init__(version=version, store_type=store_type)


class MetadataPredictionProvider(
    SerializableProviderMixin, MutableMapping[ImageId, pd.DataFrame]
):
    __store_class__ = MetadataPredictionsProviderStore

    def __init__(
        self,
        provider: Mapping[ImageId, pd.DataFrame] | pd.DataFrame | dict | None = None,
        *,
        identifier: str | None = None,
    ) -> None:
        if provider is None:
            provider = {}

        if isinstance(provider, type(self)):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            try:
                _ = list(map(ImageId.from_str, provider.index))
            except (TypeError, ValueError):
                raise ValueError("provider dataframe index has non ImageId indices")
            self.df = provider.loc[:, list(MetadataPrediction.__fields__)].copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, dict):
            if not provider:
                self.df = pd.DataFrame(columns=MetadataPrediction.__fields__)
            else:
                columns = set()
                dfs = []
                for image_id, df in provider.items():
                    if df.empty:
                        continue
                    df = df.loc[:, list(MetadataPrediction.__fields__)]
                    ids = set(df.index.unique())
                    if len(ids) > 2:
                        raise ValueError(f"image_ids in provider not unique: {ids!r}")
                    image_id_str = image_id.to_str()
                    if {image_id_str} == ids:
                        pass
                    elif {None, image_id_str}.issuperset(ids):
                        index = df.index.fillna(image_id_str)
                        df = df.set_index(index)
                    else:
                        raise AssertionError(f"{image_id_str} with Index: {ids!r}")
                    dfs.append(df)
                    columns.add(frozenset(df.columns))
                if len(columns) != 1:
                    raise RuntimeError(
                        f"dataframe columns in provider don't match {columns!r}"
                    )
                self.df = pd.concat(dfs)
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(
                f"expected `BaseMetadataProvider`, got: {type(provider).__name__!r}"
            )

    def __getitem__(self, image_id: ImageId) -> pd.DataFrame:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        return self.df.loc[[image_id.to_str()]]

    def __setitem__(self, image_id: ImageId, value: pd.DataFrame) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        if not value.columns == self.df.columns:
            raise ValueError("dataframe columns do not match")
        self.df = pd.concat(
            [
                self.df.drop(image_id.to_str()),
                value.set_index(pd.Index([image_id.to_str()] * len(value))),
            ]
        )

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        self.df.drop(image_id.to_str(), inplace=True)

    def __len__(self) -> int:
        return self.df.index.nunique(dropna=True)

    def __iter__(self) -> Iterator[ImageId]:
        return map(ImageId.from_str, self.df.index.unique())

    __repr__ = mapping_repr


class GroupedMetadataPredictionProvider(
    GroupedProviderMixin, MetadataPredictionProvider
):
    __provider_class__ = MetadataPredictionProvider


# === NOT IMPLEMENTED YET =====================================================


# noinspection PyAbstractClass
class AnnotationPredictionProvider(MutableMapping[ImageId, Any]):
    def __init__(
        self,
        provider: pd.DataFrame | dict | None = None,
        *,
        identifier: str | None = None,
    ) -> None:
        raise NotImplementedError("todo")
