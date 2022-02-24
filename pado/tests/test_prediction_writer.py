from __future__ import annotations

import os
from itertools import product
from itertools import tee

import numpy as np
import tiffslide

from pado.images.utils import Bounds
from pado.io.files import urlpathlike_to_fs_and_path
from pado.predictions.writers import ImagePredictionWriter


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def test_prediction_writer(dataset):
    """tests the prediction writer"""
    item = dataset.get_by_idx(0)

    iid = item.id
    image = item.image

    tile_size = 100
    tile_shape = (tile_size, tile_size, 5)
    tile_mpp = image.mpp
    tile_dtype = np.dtype("f8")

    writer = ImagePredictionWriter(extra_metadata={"some": "metadata"})
    writer.set_input(image_id=iid, image_size=image.dimensions)
    writer.set_output(tile_shape=tile_shape, tile_dtype=tile_dtype)

    iw, ih = image.dimensions.as_tuple()
    coords = list(product(range(0, iw, tile_size), range(0, ih, tile_size)))
    for idx, (x0, y0) in enumerate(coords):
        b = Bounds(x0, y0, x0 + tile_size, y0 + tile_size, mpp=tile_mpp)
        print(idx, b)
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
    # for the test images ts_zarr_grp is an Array
    assert set(np.unique(preds.ts_zarr_grp)) == set(range(len(coords)))
