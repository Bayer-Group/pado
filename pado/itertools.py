from __future__ import annotations

from typing import Iterator

from pado.dataset import PadoDataset
from pado.dataset import PadoItem
from pado.types import CollatedPadoItems

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object


__all__ = [
    "SlideDataset",
]


class SlideDataset(Dataset):
    """A thin wrapper around a pado dataset for data loading

    Provides map-style and iterable-style dataset interfaces
    """

    def __init__(
        self,
        ds: PadoDataset,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._ds = ds

    def __getitem__(self, index: int) -> PadoItem:
        try:
            return self._ds[index]
        except IndexError:
            raise KeyError

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


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from pado.mock import mock_dataset

    _ds = mock_dataset("memory://somewhere", num_images=97)
    sds = SlideDataset(_ds)

    loader = DataLoader(
        sds,
        batch_size=10,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=3,
        collate_fn=sds.collate_fn,
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
            print(
                idx,
                {
                    "id": len(xx["id"]),
                    "image": len(xx["image"]),
                    "metadata": len(xx["metadata"]),
                    "annotations": len(xx["annotations"]),
                },
            )

    print("start")
    consume(loader)
    print("stop")
