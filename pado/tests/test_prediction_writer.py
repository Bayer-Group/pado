from __future__ import annotations

import os
from itertools import product
from itertools import tee

import numpy as np
import tiffslide

from pado.dataset import PadoDataset
from pado.images.utils import IntBounds
from pado.io.files import urlpathlike_to_fs_and_path
from pado.predictions.writers import ImagePredictionWriter
from pado.predictions.writers import _multichannel_to_rgb


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def test_single_class_to_rgb():
    """test conversion of single channel array to an RGB array"""
    arr = np.ones((10, 10, 1))
    arr[0, 0, 0] = 0.3
    color_conversion_kwargs = {
        "channel_colors": [(255, 0, 0)],
        "single_channel_threshold": 0.5,
    }
    rgb_array = _multichannel_to_rgb(arr, **color_conversion_kwargs)
    assert rgb_array.ndim == 3 and rgb_array.shape[2] == 3
    assert len(np.unique(rgb_array.reshape(-1, 3), axis=0)) == 2


def test_multiclass_to_rgb():
    """test conversion of multi-channel array to an RGB array"""

    arr = np.zeros((10, 10, 3))

    arr[0, 0, 0] = 1

    arr[5, 5, 1] = 1

    arr[9, 9, 0] = 0.1
    arr[9, 9, 1] = 0.1
    arr[9, 9, 2] = 0.8

    color_conversion_kwargs = {
        "channel_colors": [(255, 255, 255), (255, 0, 0), (0, 255, 0)]
    }

    rgb_array = _multichannel_to_rgb(arr, **color_conversion_kwargs)

    assert rgb_array.ndim == 3 and rgb_array.shape[2] == 3

    assert (rgb_array[0, 0] == (255, 255, 255)).all()
    assert (rgb_array[5, 5] == (255, 0, 0)).all()
    assert (rgb_array[9, 9] == (0, 255, 0)).all()


def test_prediction_writer(dataset):
    """tests the prediction writer"""
    item = dataset[0]

    iid = item.id
    image = item.image

    num_channels = 5
    tile_size = 100
    tile_shape = (tile_size, tile_size, num_channels)
    tile_mpp = image.mpp
    tile_dtype = np.dtype("f8")
    output_colors = [
        (0, 0, 0),
        (64, 64, 64),
        (128, 128, 128),
        (196, 196, 196),
        (255, 255, 255),
    ]

    writer = ImagePredictionWriter(extra_metadata={"some": "metadata"})
    writer.set_input(image_id=iid, image_size=image.dimensions)
    writer.set_output(
        tile_shape=tile_shape, tile_dtype=tile_dtype, channel_colors=output_colors
    )

    iw, ih = image.dimensions.as_tuple()
    coords = list(product(range(0, iw, tile_size), range(0, ih, tile_size)))
    for idx, (x0, y0) in enumerate(coords):
        b = IntBounds(x0, y0, x0 + tile_size, y0 + tile_size, mpp=tile_mpp)
        writer.add_prediction(
            np.full(tile_shape, idx, dtype=tile_dtype),
            bounds=b,
        )

    writer.store_in_dataset(dataset, predictions_path="_my_predictions")

    # asserts
    fs, path = urlpathlike_to_fs_and_path(dataset.urlpath)
    tiffs = fs.glob(os.path.join(path, "_my_predictions/*.tif"))
    assert len(tiffs) == 1
    preds = tiffslide.TiffSlide(tiffs[0])
    assert preds.dimensions == (image.dimensions.height, image.dimensions.width)

    # assert set(np.unique(preds.ts_zarr_grp["0"])) == set(range(len(coords)))
    # ^^^ this assertion will be relevant once we support non-pre-baked rgb overlays

    ds = PadoDataset(dataset.urlpath)
    assert len(ds.predictions.images[iid]) == 1
    with ds.predictions.images[iid][0].image as pred_image:
        assert pred_image.level_count > 1
