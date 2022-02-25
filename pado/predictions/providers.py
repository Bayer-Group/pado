from __future__ import annotations

import enum
import json
from typing import Any

import pandas as pd

from pado.collections import GroupedProviderMixin
from pado.collections import PadoMutableSequence
from pado.collections import PadoMutableSequenceMapping
from pado.collections import ProviderStoreMixin
from pado.collections import SerializableProviderMixin
from pado.images import Image
from pado.images import ImageId
from pado.images.utils import Bounds
from pado.io.store import Store
from pado.io.store import StoreType


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
        assert store_type == StoreType.IMAGE_PREDICTIONS
        super().__init__(version=version, store_type=store_type)


class ImagePredictionProvider(
    PadoMutableSequenceMapping[ImagePredictions], SerializableProviderMixin
):
    __store_class__ = ImagePredictionsProviderStore
    __value_class__ = ImagePredictions


class GroupedImagePredictionProvider(GroupedProviderMixin, ImagePredictionProvider):
    __provider_class__ = ImagePredictionProvider


# === NOT IMPLEMENTED YET =====================================================


class AnnotationPredictionProvider:
    def __init__(
        self,
        provider: pd.DataFrame | dict | None = None,
        *,
        identifier: str | None = None,
    ) -> None:
        raise NotImplementedError("todo")


class MetadataPredictionProvider:
    def __init__(
        self,
        provider: pd.DataFrame | dict | None = None,
        *,
        identifier: str | None = None,
    ) -> None:
        raise NotImplementedError("todo")
