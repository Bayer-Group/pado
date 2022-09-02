from __future__ import annotations

import os
import sys
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

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

import numpy as np
import orjson
from shapely.geometry import box
from tqdm import tqdm

from pado.annotations.annotation import AnnotationIndex
from pado.annotations.annotation import ensure_validity
from pado.annotations.annotation import scale_annotation
from pado.annotations.annotation import translate_annotation
from pado.dataset import PadoItem
from pado.images.ids import ImageId
from pado.images.tiles import PadoTileItem
from pado.images.tiles import TileId
from pado.images.tiles import TileIndex
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

    from pado.annotations import Annotation  # noqa
    from pado.dataset import PadoDataset
    from pado.images.tiles import TilingStrategy
    from pado.images.utils import MPP
    from pado.images.utils import IntSize  # noqa


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


def ensure_callable(func: Any) -> Callable[..., Any] | None:
    if func is None:
        return None
    elif not callable(func):
        raise ValueError(f"{func!r} not callable")
    else:
        return func


TransformCallable: TypeAlias = "Callable[[PadoTileItem], PadoTileItem]"
ErrorHandlerCallable: TypeAlias = "Callable[[TileDataset, int, BaseException], bool]"
AnnotationsMaskMapper: TypeAlias = "Callable[[list[Annotation], IntSize], NDArray]"
AnnotationsPolyMapper: TypeAlias = "Callable[[Annotation], Annotation | None]"


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
        transform: TransformCallable | None = None,
        as_tensor: bool = True,
        image_storage_options: dict[str, Any] | None = None,
        error_handler: ErrorHandlerCallable | None = None,
        annotations_crop: bool = False,
        annotations_mpp_scale: bool = False,
        annotations_mask_mapper: AnnotationsMaskMapper | None = None,
        annotations_poly_mapper: AnnotationsPolyMapper | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ds = ds
        self._ts = tiling_strategy
        self._strategy_str = self._ts.serialize()
        self._precompute_kw = precompute_kw or {}
        self._transform = ensure_callable(transform)
        self._as_tensor = bool(as_tensor)
        self._image_so = image_storage_options
        self._error_handler = error_handler or (lambda td, idx, exc: False)
        self._annotations_crop = bool(annotations_crop)
        self._annotations_mpp_scale = bool(annotations_mpp_scale)
        self._annotations_mask_mapper = ensure_callable(annotations_mask_mapper)
        self._annotations_poly_mapper = ensure_callable(annotations_poly_mapper)

        # internal caches
        self._cumulative_num_tiles: NDArray[np.int64] | None = None
        self._tile_indexes: dict[ImageId, TileIndex] = {}
        self._annotation_trees: dict[ImageId, AnnotationIndex | None] = {}

    def precompute_tiling(
        self,
        workers: int | None = None,
        *,
        force: bool = False,
    ) -> bool:
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

                image_ids = set(self._ds.annotations).intersection(self._ds.index)
                if not force:
                    image_ids -= set(self._annotation_trees)

                for image_id in tqdm(image_ids, desc="precomputing annotation trees"):
                    _annotations = self._ds.annotations[image_id]
                    self._annotation_trees[image_id] = AnnotationIndex.from_annotations(
                        _annotations
                    )

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
                                AnnotationIndex.from_annotations,
                                (self._ds.annotations[iid] for iid in image_ids),
                            ),
                        ),
                        desc="precomputing annotation trees",
                    ):
                        self._annotation_trees[image_id] = str_tree

            self._cumulative_num_tiles = np.cumsum(
                [len(self._tile_indexes[iid]) for iid in self._ds.index], dtype=np.int64
            )
        return False

    def requires_precompute(self) -> bool:
        """return if the TileIndex would need to precompute"""
        compute_tile_indexes = (
            self._cumulative_num_tiles is None
            or len(self._cumulative_num_tiles) != len(self._ds.index)
            or not set(self._ds.index).issubset(self._tile_indexes)
        )
        if compute_tile_indexes:
            return True
        compute_annotation_indexes = not set(self._ds.annotations).issubset(
            self._annotation_trees
        )
        if compute_annotation_indexes:
            return True
        # we passed the check! now let's speed this up
        self.__dict__["requires_precompute"] = lambda: False
        return False

    def __getitem__(self, index: int) -> PadoTileItem:
        if self.requires_precompute():
            self.precompute_tiling(**self._precompute_kw)
        while True:

            try:
                if index < 0:
                    raise NotImplementedError(index)

                # retrieve the pado item
                slide_idx = int(
                    np.searchsorted(
                        self._cumulative_num_tiles, index, side="right", sorter=None
                    )
                )
                pado_item = self._ds[slide_idx]

                # get the tile location
                tile_index = self._tile_indexes[pado_item.id]
                if slide_idx > 0:
                    idx = index - self._cumulative_num_tiles[slide_idx - 1]
                else:
                    idx = index
                location, size, mpp = tile_index[idx]

                # get the tile array
                with pado_item.image.via(
                    self._ds, storage_options=self._image_so
                ) as img:
                    lvl0_mpp = img.mpp
                    arr = img.get_array_at_mpp(location, size, target_mpp=mpp)
                if self._as_tensor:
                    arr = from_numpy(arr)

                # filter the annotations
                slide_annotations = pado_item.annotations
                str_tree = self._annotation_trees.get(pado_item.id, None)
                if str_tree is not None:
                    x0, y0 = location.scale(lvl0_mpp).as_tuple()
                    tw, th = size.scale(lvl0_mpp).as_tuple()
                    tile_box = box(x0, y0, x0 + tw, y0 + th)
                    idxs = str_tree.query_items(tile_box)
                    tile_annotations = [slide_annotations[i] for i in idxs]
                else:
                    tile_box = None
                    tile_annotations = None

                # postprocess annotations
                if tile_annotations:
                    for a_idx, a in zip(
                        reversed(range(len(tile_annotations))),
                        reversed(tile_annotations),
                    ):
                        a.__dict__["_readonly"] = False
                        if self._annotations_poly_mapper:
                            a = self._annotations_poly_mapper(a)
                            if a is None:
                                tile_annotations.pop(a_idx)
                                continue
                        a = ensure_validity(a)
                        if self._annotations_crop:
                            a.geometry = tile_box.intersection(a.geometry)
                        if self._annotations_mpp_scale:
                            a = scale_annotation(a, level0_mpp=lvl0_mpp, target_mpp=mpp)

                        a = translate_annotation(a, location=location)
                        a._readonly = True
                        tile_annotations[a_idx] = a

                # convert to mask if wanted
                if self._annotations_mask_mapper:
                    tile_annotations = self._annotations_mask_mapper(
                        tile_annotations, size
                    )

                # create the tile item
                tile_item = PadoTileItem(
                    id=TileId(
                        image_id=pado_item.id, strategy=self._strategy_str, index=idx
                    ),
                    tile=arr,
                    metadata=pado_item.metadata,
                    annotations=tile_annotations,
                )

                # apply user transforms
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
                    raise

    def __iter__(self) -> Iterator[PadoTileItem]:
        # can be done faster by lazy evaluating the len of slides
        for idx in range(0, len(self)):
            yield self[idx]

    def __len__(self):
        if self.requires_precompute():
            self.precompute_tiling(**self._precompute_kw)
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
            "ds": {
                "urlpath": self._ds.urlpath,
                "storage_options": self._ds.storage_options,
            },
            "tiling_strategy": self._strategy_str,
            "caches": {
                "tile_indexes": tile_indexes,
                "annotation_trees": annotation_trees,
            },
        }
        for iid, tile_index in self._tile_indexes.items():
            tile_indexes[iid.to_str()] = tile_index.to_json(as_string=False)
        for iid, str_tree in self._annotation_trees.items():
            annotation_trees[iid.to_str()] = str_tree.to_json(as_string=False)

        with open(fn, mode="wb") as f:
            f.write(
                orjson.dumps(
                    dct, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
                )
            )
        return dct

    def caches_load(self, fn: os.PathLike | str) -> dict:
        """load caches from disk"""
        with open(fn, mode="rb") as f:
            dct = orjson.loads(f.read())

        if dct["type"] != "pado.itertools.TileDataset:cache":
            raise ValueError("incorrect type cache json")

        if "tiling_strategy" in dct:
            if dct["tiling_strategy"] != self._strategy_str:
                raise ValueError(
                    "caches are for a different tiling strategy: "
                    f"{dct['tiling_strategy']!r} != {self._strategy_str!r}"
                )

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
                ImageId.from_str(key): AnnotationIndex.from_json(value)
                for key, value in annotation_trees_json.items()
            }
        )
        return dct


# === helpers =================================================================


def iter_exc_chain(exc: BaseException | None) -> Generator[BaseException]:
    if exc:
        yield exc
        if isinstance(exc, BaseException):
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
        self._num_retries = int(num_retries) if num_retries is not None else None
        self._retry_delay = float(retry_delay)
        self._total_delay = float(total_delay) if total_delay is not None else None
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
