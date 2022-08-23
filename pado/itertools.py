from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from typing import Callable
from typing import Iterator

import numpy as np
from tqdm import tqdm

from pado.dataset import PadoItem
from pado.images.tiles import PadoTileItem
from pado.images.tiles import TileId
from pado.images.tiles import TileIndex
from pado.images.tiles import TilingStrategy
from pado.types import CollatedPadoItems
from pado.types import CollatedPadoTileItems

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pado.dataset import PadoDataset
    from pado.images import ImageId
    from pado.images.utils import MPP


__all__ = [
    "SlideDataset",
]


# === iterate over slides =====================================================


class SlideDataset(Dataset):
    """A thin wrapper around a pado dataset for data loading

    Provides map-style and iterable-style dataset interfaces
    for loading entire slides with metadata and annotations
    """

    def __init__(
        self,
        ds: PadoDataset,
        *,
        transform: Callable[[PadoItem], PadoItem] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ds = ds
        if transform is None:
            self._transform = None
        elif not callable(transform):
            raise ValueError("transform not callable")
        else:
            self._transform = transform

    def __getitem__(self, index: int) -> PadoItem:
        try:
            item = self._ds[index]
        except IndexError:
            raise KeyError

        if self._transform:
            item = self._transform(item)
        return item

    def __iter__(self) -> Iterator[PadoItem]:
        for iid in self._ds.index:
            yield self._ds[iid]

    def __len__(self):
        return len(self._ds)

    @staticmethod
    def collate_fn(batch: list[PadoItem]) -> CollatedPadoItems:
        it = zip(PadoItem._fields, map(list, zip(*batch)))
        # noinspection PyArgumentList
        return CollatedPadoItems(it)


# === iterate over tiles ======================================================


class TileDataset(Dataset):
    """A thin wrapper around a pado dataset for data loading

    Provides map-style and iterable-style dataset interfaces
    for loading entire tiles with metadata and annotations
    """

    def __init__(
        self,
        ds: PadoDataset,
        *,
        tiling_strategy: TilingStrategy,
        precompute_kw: dict | None = None,
        transform: Callable[[PadoTileItem], PadoTileItem] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ds = ds
        self._ts = tiling_strategy
        self._precompute_kw = precompute_kw or {}
        self._strategy_str = self._ts.serialize()
        self._cumulative_num_tiles: NDArray[np.int64] | None = None
        self._tile_indexes: dict[ImageId, TileIndex] = {}
        if transform is None:
            self._transform = None
        elif not callable(transform):
            raise ValueError("transform not callable")
        else:
            self._transform = transform

    def precompute_tiling(self, workers: int | None = None):
        if self._cumulative_num_tiles is None and not self._tile_indexes:
            if workers is None:
                for image_id in tqdm(self._ds.index, desc="precomputing tile indices"):
                    image = self._ds.images[image_id]
                    self._tile_indexes[image_id] = self._ts.precompute(image)

            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for image_id, tile_index in tqdm(
                        zip(
                            self._ds.index,
                            executor.map(self._ts.precompute, self._ds.images.values()),
                        ),
                        desc="precomputing tile indices",
                    ):
                        self._tile_indexes[image_id] = tile_index

            self._cumulative_num_tiles = np.cumsum(
                [len(self._tile_indexes[iid]) for iid in self._ds.index], dtype=np.int64
            )
        return

    def _ensure_precompute(self):
        if self._cumulative_num_tiles is None and not self._tile_indexes:
            self.precompute_tiling(**self._precompute_kw)

    def __getitem__(self, index: int) -> PadoTileItem:
        self._ensure_precompute()
        if index < 0:
            raise NotImplementedError(index)
        slide_idx = int(
            np.searchsorted(
                self._cumulative_num_tiles, index, side="right", sorter=None
            )
        )
        pado_item = self._ds[slide_idx]

        tile_index = self._tile_indexes[pado_item.id]
        if slide_idx > 0:
            idx = index - self._cumulative_num_tiles[slide_idx - 1]
        else:
            idx = index
        location, size, mpp = tile_index[idx]

        with pado_item.image.via(self._ds) as img:
            arr = img.get_array_at_mpp(location, size, target_mpp=mpp)

        tile_item = PadoTileItem(
            id=TileId(image_id=pado_item.id, strategy=self._strategy_str, index=idx),
            tile=arr,
            metadata=pado_item.metadata,
            annotations=pado_item.annotations,
        )
        if self._transform:
            tile_item = self._transform(tile_item)
        return tile_item

    def __iter__(self) -> Iterator[PadoTileItem]:
        # can be done faster by lazy evaluating the len of slides
        for idx in range(0, len(self)):
            yield self[idx]

    def __len__(self):
        self._ensure_precompute()
        return self._cumulative_num_tiles[-1]

    @staticmethod
    def collate_fn(batch: list[PadoTileItem]) -> CollatedPadoTileItems:
        it = zip(PadoTileItem._fields, map(list, zip(*batch)))
        # noinspection PyArgumentList
        return CollatedPadoTileItems(it)


if __name__ == "__main__":
    import sys

    from torch.utils.data import DataLoader

    from pado.mock import mock_dataset

    _ds = mock_dataset("memory://somewhere", num_images=97)

    if sys.argv[1] == "slide":
        dataset = SlideDataset(_ds)

    elif sys.argv[1] == "tile":
        from pado.images.tiles import FastGridTiling
        from pado.images.utils import MPP  # noqa

        dataset = TileDataset(
            _ds,
            tiling_strategy=FastGridTiling(
                tile_size=(10, 10),
                target_mpp=MPP(1, 1),
                overlap=0,
                min_chunk_size=0.0,  # use 0.2 or so with real data
                normalize_chunk_sizes=True,
            ),
        )

    else:
        print("argv[1] != 'slide' or 'tile'")
        raise SystemExit(1)

    print("total length:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=3,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=2,
        persistent_workers=False,
    )

    def consume(x):
        for idx, xx in enumerate(x):
            print(idx, {k: len(v) for k, v in xx.items()})

    print("start")
    consume(loader)
    print("stop")
