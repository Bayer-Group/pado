from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from logging import getLogger
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterator

import numpy as np
import orjson
from shapely.strtree import STRtree
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm

from pado.dataset import PadoItem
from pado.images.tiles import PadoTileItem
from pado.images.tiles import TileId
from pado.images.tiles import TileIndex
from pado.images.tiles import TilingStrategy
from pado.types import CollatedPadoItems
from pado.types import CollatedPadoTileItems

try:
    from torch import from_numpy
    from torch import stack
    from torch.utils.data import Dataset
except ImportError:
    stack = None
    Dataset = object

    def from_numpy(x):
        return x


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pado.annotations import Annotations
    from pado.dataset import PadoDataset
    from pado.images.ids import ImageId
    from pado.images.utils import MPP


__all__ = [
    "SlideDataset",
    "TileDataset",
    "RetryErrorHandler",
]

_log = getLogger("pado.itertools")

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


def call_precompute(ts, args, kwargs):
    return ts.precompute(*args, **kwargs)


def build_str_tree(annotations: Annotations) -> STRtree | None:
    if annotations:
        geometries = [a.geometry for a in annotations]
        return STRtree(geometries)
    else:
        return None


def load_annotation_tree(obj: dict | None) -> STRtree | None:
    if obj is None:
        return None

    t = obj["type"]
    if t != "shapely.strtree.STRtree":
        raise NotImplementedError(t)
    geometries = obj["geometries"]
    return STRtree([wkt_loads(o) for o in geometries])


def dump_annotation_tree(obj: STRtree | None) -> dict | None:
    if obj is None:
        return None

    if hasattr(obj, "_geoms"):
        # shapely < 2.0.0
        geometries = getattr(obj, "_geoms")
    else:
        # shapely >= 2.0.0
        geometries = getattr(obj, "geometries")
    return {
        "type": "shapely.strtree.STRtree",
        "geometries": [o.wkt for o in geometries],
    }


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
        as_tensor: bool = True,
        image_storage_options: dict[str, Any] | None = None,
        error_handler: Callable[[TileDataset, int, BaseException], bool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ds = ds
        self._ts = tiling_strategy
        self._precompute_kw = precompute_kw or {}
        self._strategy_str = self._ts.serialize()
        self._cumulative_num_tiles: NDArray[np.int64] | None = None
        self._tile_indexes: dict[ImageId, TileIndex] = {}
        self._annotation_trees: dict[ImageId, STRtree | None] = {}
        self._error_handler = error_handler or (lambda td, idx, exc: False)

        if transform is None:
            self._transform = None
        elif not callable(transform):
            raise ValueError("transform not callable")
        else:
            self._transform = transform
        self._as_tensor = bool(as_tensor)
        self._image_so = image_storage_options

    def precompute_tiling(self, workers: int | None = None, *, force: bool = False):
        compute_tile_indexes = (
            self._cumulative_num_tiles is None
            or len(self._cumulative_num_tiles) != len(self._ds.index)
            or not set(self._ds.index).issubset(self._tile_indexes)
        )
        compute_annotation_indexes = not set(self._ds.annotations).issubset(
            self._annotation_trees
        )

        if compute_tile_indexes or compute_annotation_indexes or force:

            if workers is None:
                # single worker sequential
                image_ids = set(self._ds.index)
                if not force:
                    image_ids -= set(self._tile_indexes)

                for image_id in tqdm(image_ids, desc="precomputing tile indices"):
                    image = self._ds.images[image_id]
                    self._tile_indexes[image_id] = call_precompute(
                        self._ts, (image,), {"storage_options": self._image_so}
                    )

                image_ids = set(self._ds.annotations)
                if not force:
                    image_ids -= set(self._annotation_trees)

                for image_id in tqdm(image_ids, desc="precomputing annotation trees"):
                    _annotations = self._ds.annotations[image_id]
                    self._annotation_trees[image_id] = build_str_tree(_annotations)

            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    image_ids = set(self._ds.index)
                    if not force:
                        image_ids -= set(self._tile_indexes)

                    for image_id, tile_index in tqdm(
                        zip(
                            image_ids,
                            executor.map(
                                call_precompute,
                                repeat(self._ts),
                                ((self._ds.images[iid],) for iid in image_ids),
                                repeat({"storage_options": self._image_so}),
                            ),
                        ),
                        desc="precomputing tile indices",
                    ):
                        self._tile_indexes[image_id] = tile_index

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    image_ids = set(self._ds.annotations)
                    if not force:
                        image_ids -= set(self._annotation_trees)

                    for image_id, str_tree in tqdm(
                        zip(
                            image_ids,
                            executor.map(
                                build_str_tree,
                                (self._ds.annotations[iid] for iid in image_ids),
                            ),
                        ),
                        desc="precomputing annotation trees",
                    ):
                        self._annotation_trees[image_id] = str_tree

            self._cumulative_num_tiles = np.cumsum(
                [len(self._tile_indexes[iid]) for iid in self._ds.index], dtype=np.int64
            )
        return

    def _ensure_precompute(self):
        if self._cumulative_num_tiles is None and not self._tile_indexes:
            self.precompute_tiling(**self._precompute_kw)

    def __getitem__(self, index: int) -> PadoTileItem:
        while True:

            try:
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

                with pado_item.image.via(
                    self._ds, storage_options=self._image_so
                ) as img:
                    arr = img.get_array_at_mpp(location, size, target_mpp=mpp)

                if self._as_tensor:
                    arr = from_numpy(arr)

                tile_item = PadoTileItem(
                    id=TileId(
                        image_id=pado_item.id, strategy=self._strategy_str, index=idx
                    ),
                    tile=arr,
                    metadata=pado_item.metadata,
                    annotations=pado_item.annotations,
                )
                if self._transform:
                    tile_item = self._transform(tile_item)
                return tile_item
            except KeyboardInterrupt:
                raise

            except BaseException as e:
                should_retry = self._error_handler(self, index, e)
                if should_retry:
                    _log.exception(f"Retrying {self!r}[{index}] ...")
                    continue
                else:
                    raise e

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
        dct = CollatedPadoTileItems(it)
        tile = dct["tile"]
        # collate tiles
        if tile:
            if isinstance(tile[0], np.ndarray):
                dct["tile"] = np.stack(tile)
            else:
                dct["tile"] = stack(tile)
        return dct

    def caches_dump(self, fn: os.PathLike | str) -> dict:
        """dump caches to disk"""
        tile_indexes = {}
        annotation_trees = {}
        dct = {
            "type": "pado.itertools.TileDataset:cache",
            "version": 1,
            "ds_urlpath": self._ds.urlpath,
            "caches": {
                "tile_indexes": tile_indexes,
                "annotation_trees": annotation_trees,
            },
        }
        for iid, tile_index in self._tile_indexes.items():
            tile_indexes[iid.to_str()] = tile_index.to_json(as_string=False)
        for iid, str_tree in self._annotation_trees.items():
            annotation_trees[iid.to_str()] = dump_annotation_tree(str_tree)

        with open(fn, mode="wb") as f:
            f.write(orjson.dumps(dct))
        return dct

    def caches_load(self, fn: os.PathLike | str) -> dict:
        """load caches from disk"""
        with open(fn, mode="rb") as f:
            dct = orjson.loads(f.read())

        if dct["type"] != "pado.itertools.TileDataset:cache":
            raise ValueError("incorrect type cache json")

        tile_indexes_json = dct["caches"]["tile_indexes"]
        annotation_trees_json = dct["caches"]["annotation_trees"]

        self._tile_indexes.update(
            {
                ImageId.from_str(key): TileIndex.from_json(value)
                for key, value in tile_indexes_json.items()
            }
        )
        self._annotation_trees.update(
            {
                ImageId.from_str(key): load_annotation_tree(value)
                for key, value in annotation_trees_json.items()
            }
        )
        return dct


# === helpers =================================================================


def iter_exc_chain(exc: BaseException | None) -> Generator[BaseException]:
    if exc:
        yield exc
        yield from iter_exc_chain(exc.__cause__ or exc.__context__)


class RetryErrorHandler:
    def __init__(
        self,
        exception_type: tuple[type[BaseException], ...] | type[BaseException],
        *,
        retry_delay: float = 0.1,
        num_retries: int | None = None,
        total_delay: float | None = None,
        exponential_backoff: bool = False,
        check_exception_chain: bool = False,
    ) -> None:
        """a retry error handler

        Parameters
        ----------
        exception_type:
            the exception classes which should be used for retrying
        retry_delay:
            the retry wait delay in seconds
        num_retries:
            the maximum amount of retries (infinite if `None`)
        total_delay:
            the maximum total delay added via retries in seconds
        exponential_backoff:
            makes the n-th delay wait retry_delay * 2**n seconds.
            By default, each delay waits retry_delay.
        check_exception_chain:
            also check the __cause__ and __context__ chain of the
            exception and match if any item in the chain is a match.

        """
        if isinstance(exception_type, type) and issubclass(
            exception_type, BaseException
        ):
            exception_type = (exception_type,)
        if not isinstance(exception_type, tuple):
            raise TypeError(
                f"expected tuple[type[BaseException]], got {type(exception_type).__name__!r}"
            )
        if not all(issubclass(e, BaseException) for e in exception_type):
            _types = tuple(type(e).__name__ for e in exception_type)
            raise TypeError(f"expected tuple[type[BaseException]], got {_types}")
        if num_retries is None and total_delay is None:
            raise ValueError("must provide one of `num_retries` or `timeout_sec`")
        self._exception_type = exception_type
        self._num_retries = int(num_retries) if num_retries else None
        self._retry_delay = float(retry_delay)
        self._total_delay = float(total_delay) if total_delay else None
        self._exp_backoff = bool(exponential_backoff)
        self._check_exc_chain = bool(check_exception_chain)
        self._call_counter = Counter()
        self._sleep = time.sleep

    def __call__(self, td: TileDataset, index: int, exception: BaseException) -> bool:
        """return if the action should be retried"""
        if index not in self._call_counter:
            # we reset the counter if a new index is requested
            self._call_counter.clear()

        # get sleep delays
        call_cnt = n = self._call_counter[index]
        if self._exp_backoff:
            current_sleep = self._retry_delay * 2**n
            total_sleep = self._retry_delay * (2 ** (n + 1) - 1)
        else:
            current_sleep = self._retry_delay
            total_sleep = self._retry_delay * (n + 1)

        if self._check_exc_chain:
            matches = any(
                isinstance(e, self._exception_type) for e in iter_exc_chain(exception)
            )
        else:
            matches = isinstance(exception, self._exception_type)

        # check if we should retry
        if (
            matches
            and (self._num_retries is None or call_cnt < self._num_retries)
            and (self._total_delay is None or total_sleep < self._total_delay)
        ):
            self._sleep(current_sleep)
            self._call_counter[index] += 1
            return True
        else:
            return False


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
            error_handler=RetryErrorHandler(
                TimeoutError,
                retry_delay=0.1,
                total_delay=30.0,
                exponential_backoff=True,
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
