""""""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Optional

import pandas as pd

from pado.io.files import fsopen
from pado.predictions.providers import AnnotationPredictionProvider
from pado.predictions.providers import GroupedImagePredictionProvider
from pado.predictions.providers import GroupedMetadataPredictionProvider
from pado.predictions.providers import ImagePredictionProvider
from pado.predictions.providers import ImagePredictions
from pado.predictions.providers import MetadataPredictionProvider

if TYPE_CHECKING:
    from pado.dataset import PadoDataset
    from pado.images.ids import ImageId

__all__ = [
    "PredictionProxy",
]


class PredictionProxy:
    def __init__(self, ds: PadoDataset):
        self._ds: PadoDataset = ds

        # caches
        self._images: ImagePredictionProvider | None = None
        self._annotations: AnnotationPredictionProvider | None = None
        self._metadata: MetadataPredictionProvider | None = None

    # === data ===

    @property
    def images(self) -> ImagePredictionProvider:
        if self._images is None:
            # noinspection PyProtectedMember
            fs, get_fspath = self._ds._fs, self._ds._get_fspath
            providers = [
                ImagePredictionProvider.from_parquet(fsopen(fs, p, mode="rb"))
                for p in fs.glob(get_fspath("*.image_predictions.parquet"))
                if fs.isfile(p)
            ]

            if len(providers) == 0:
                provider = ImagePredictionProvider()
            elif len(providers) == 1:
                provider = providers[0]
            else:
                provider = GroupedImagePredictionProvider(*providers)

            self._images = provider
        return self._images

    @property
    def annotations(self) -> AnnotationPredictionProvider:
        # noinspection PydanticTypeChecker,PyTypeChecker
        return {}  # fixme: todo

    @property
    def metadata(self) -> MetadataPredictionProvider:
        if self._metadata is None:
            # noinspection PyProtectedMember
            fs, get_fspath = self._ds._fs, self._ds._get_fspath
            providers = [
                MetadataPredictionProvider.from_parquet(fsopen(fs, p, mode="rb"))
                for p in fs.glob(get_fspath("*.metadata_predictions.parquet"))
                if fs.isfile(p)
            ]

            if len(providers) == 0:
                provider = MetadataPredictionProvider()
            elif len(providers) == 1:
                provider = providers[0]
            else:
                provider = GroupedMetadataPredictionProvider(*providers)

            self._metadata = provider
        return self._metadata

    # === access ===

    def get_by_id(self, image_id: ImageId) -> PadoPredictionItem:
        return PadoPredictionItem(
            image_id,
            self.images.get(image_id),
            self.annotations.get(image_id),
            self.metadata.get(image_id),
        )

    def get_by_idx(self, idx: int) -> PadoPredictionItem:
        iid = self._ds.index[idx]
        return self.get_by_id(iid)


class PadoPredictionItem(NamedTuple):
    id: ImageId
    image: Optional[ImagePredictions]
    annotations: Any  # fixme
    metadata: Optional[pd.DataFrame]
