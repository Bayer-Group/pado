from __future__ import annotations

from typing import Iterator

from pado.dataset import PadoDataset
from pado.dataset import PadoItem


class SlideDataset:
    def __init__(
        self,
        ds: PadoDataset,
    ):
        """iterate over slides in a pado dataset"""
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
    def collate_fn(batch):
        collated = {
            "image": [],
            "metadata": [],
            "annotations": [],
        }
        for x in batch:
            collated["image"].append(x.image)
            collated["metadata"].append(x.image)
            collated["annotations"].append(x.image)
        return collated


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset

    from pado.mock import mock_dataset

    class TorchSlideDataset(SlideDataset, Dataset):
        pass

    ds = mock_dataset("memory://somewhere", num_images=100)
    sds = TorchSlideDataset(ds)

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
                    "image": len(xx["image"]),
                    "metadata": len(xx["metadata"]),
                    "annotations": len(xx["annotations"]),
                },
            )

    print("start")
    consume(loader)
    print("stop")
