from __future__ import annotations

import warnings

import pytest

from pado.dataset import PadoItem
from pado.images.tiles import FastGridTiling
from pado.images.tiles import PadoTileItem
from pado.images.utils import MPP
from pado.itertools import SlideDataset
from pado.itertools import TileDataset
from pado.mock import mock_dataset


@pytest.fixture
def ds_iter():
    yield mock_dataset(None, num_images=7)


def test_slide_dataset(ds_iter):
    slide_ds = SlideDataset(ds_iter)

    for idx in range(len(slide_ds)):
        item = slide_ds[idx]
        assert isinstance(item, PadoItem)
        assert item.id is not None
        assert item.image is not None


def test_tile_dataset(ds_iter):
    tile_ds = TileDataset(
        ds_iter,
        tiling_strategy=FastGridTiling(
            tile_size=(10, 10),
            target_mpp=MPP(1, 1),
            overlap=0,
            min_chunk_size=0.0,  # use 0.2 or so with real data
            normalize_chunk_sizes=True,
        ),
    )

    with warnings.catch_warnings(record=True) as rec:
        tile_ds.precompute_tiling()
        # check warnings
        for w in rec:
            assert "all chunksizes identical" in str(w.message)

    for idx in range(len(tile_ds)):
        item = tile_ds[idx]
        assert isinstance(item, PadoTileItem)
        assert item.id is not None
        assert item.tile is not None
        assert item.tile.sum() > 0
