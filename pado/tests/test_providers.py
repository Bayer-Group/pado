from __future__ import annotations

import itertools
import os
import unittest.mock
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from pydantic.color import Color
from shapely.geometry import Polygon

from pado.annotations import Annotation
from pado.annotations import AnnotationProvider
from pado.annotations import Annotations
from pado.annotations import AnnotationState
from pado.annotations import Annotator
from pado.annotations import AnnotatorType
from pado.create import create_image_provider
from pado.dataset import PadoDataset
from pado.images.ids import ImageId
from pado.images.providers import ImageProvider
from pado.images.providers import copy_image
from pado.images.utils import IntBounds
from pado.io.files import find_files
from pado.io.paths import match_partial_paths_reversed
from pado.metadata.providers import MetadataProvider
from pado.mock import temporary_mock_svs
from pado.predictions.providers import ImagePredictionProvider
from pado.predictions.writers import ImagePredictionWriter


@pytest.fixture
def multi_image_folder(tmp_path):
    """prepare a folder structure with images"""
    base = tmp_path.joinpath("base_images")
    base.mkdir()
    for idx, subfolder in enumerate(["a_1_x", "b_2_y", "c_3_z"]):
        s = base.joinpath(subfolder)
        s.mkdir()
        with temporary_mock_svs(f"image_{idx}") as img_fn:
            data = Path(img_fn).read_bytes()
        s.joinpath(f"image_{idx}.svs").write_bytes(data)
    yield base


@pytest.fixture
def image_provider(multi_image_folder):
    yield create_image_provider(
        search_urlpath=multi_image_folder,
        search_glob="**/*.svs",
        output_urlpath=None,
    )


def test_create_image_provider(multi_image_folder):
    ip = create_image_provider(
        search_urlpath=multi_image_folder,
        search_glob="**/*.svs",
        output_urlpath=None,
    )
    assert len(ip) == 3


def test_create_image_provider_empty_urlpath(tmpdir):
    ip = create_image_provider(
        search_urlpath=tmpdir,
        search_glob="**/*.svs",
        output_urlpath=None,
    )

    assert len(ip) == 0


def test_write_image_provider(tmp_path, image_provider):
    out = tmp_path.joinpath("images.parquet")
    image_provider.to_parquet(out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_roundtrip_image_provider(image_provider):
    ip0 = image_provider
    with TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, "ip0.parquet")
        ip0.to_parquet(fn)
        ip1 = ImageProvider.from_parquet(fn)

    assert ip0.identifier == ip1.identifier
    assert set(ip0) == set(ip1)
    assert list(ip0.values()) == list(ip1.values())


def test_match_partial_paths_reversed_does_not_instantiate(
    multi_image_folder, image_provider
):
    # we want this test to fail whenever fsspec.spec.AbstractFileSystem or any of its
    # subclasses gets instantiated. The problem here is, that AbsractFileSystem instances
    # are cached through some metaclass magic in fsspec.spec._Cached.
    # so the way to make sure we fail is to mock __call__ in the metaclass:
    with unittest.mock.patch("fsspec.spec._Cached.__call__", side_effect=RuntimeError):
        m = match_partial_paths_reversed(
            current_urlpaths=image_provider.df.urlpath.values,
            new_urlpaths=list(multi_image_folder.rglob("*.svs")),
        )
        assert len(m) == 3


def test_copy_image(tmp_path, image_provider):
    new_dst = tmp_path.joinpath("new_storage_location")
    iid = next(iter(image_provider))

    copy_image(image_provider, iid, new_dst)

    assert "new_storage_location" in image_provider[iid].urlpath
    assert len(list(find_files(new_dst, glob="**/*.svs"))) == 1


PREDICTIONS_PER_IMAGE = 2


@pytest.fixture
def dataset_with_predictions(dataset):

    tile_size = 100
    tile_shape = (tile_size, tile_size, 3)
    tile_dtype = np.dtype("u1")
    channel_colors = [
        (0, 0, 0),
        (128, 128, 128),
        (255, 255, 255),
    ]

    for iidx, iid in enumerate(dataset.index):
        image = dataset.images[iid]

        tile_mpp = image.mpp

        for pidx in range(PREDICTIONS_PER_IMAGE):
            writer = ImagePredictionWriter(extra_metadata={"idx": iidx, "pidx": pidx})
            writer.set_input(image_id=iid, image_size=image.dimensions)
            writer.set_output(
                tile_shape=tile_shape,
                tile_dtype=tile_dtype,
                channel_colors=channel_colors,
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

    yield PadoDataset(dataset.urlpath, mode="r")


def test_grouped_image_predictions_provider(dataset_with_predictions):
    ds = dataset_with_predictions
    assert len(ds.predictions.images) == len(ds.images)

    pitem = ds.predictions.get_by_idx(0)
    image_predictions = pitem.image
    assert len(image_predictions) == PREDICTIONS_PER_IMAGE
    ipred = image_predictions[0]

    assert ipred.extra_metadata
    assert ipred.image.dimensions


def test_annotation_provider_df_access_after_update(dataset_ro):
    ds = PadoDataset(None)
    iid = dataset_ro.index[0]

    p_shape_before = ds.annotations.df.shape

    ds.annotations[iid] = a = Annotations()
    a_shape_before = a.df.shape

    a.append(
        Annotation.from_obj(
            dict(
                image_id=iid,
                project="tggates",
                annotator=Annotator(type=AnnotatorType("model"), name="test"),
                state=AnnotationState(3),
                classification="colored_square",
                color=Color((0, 0, 0, 1)),
                comment="No comment",
                geometry=Polygon.from_bounds(0, 0, 500, 500),
            )
        )
    )

    p_shape_after = ds.annotations.df.shape
    a_shape_after = a.df.shape

    assert a_shape_before != a_shape_after
    assert p_shape_before != p_shape_after


def test_empty_metadata_provider():
    mp = MetadataProvider({})
    assert len(mp) == 0


@pytest.mark.parametrize(
    "provider_cls,index0",
    itertools.product(
        [
            ImageProvider,
            MetadataProvider,
            AnnotationProvider,
            ImagePredictionProvider,
        ],
        [
            ImageId("1.svs"),
            123,
            "somestring",
        ],
    ),
    ids=lambda x: x.__name__ if isinstance(x, type) else type(x).__name__,
)
def test_provider_rejects_non_image_id_keys(provider_cls, index0):
    df = pd.DataFrame(index=[index0], data={"something": [1]})
    with pytest.raises(ValueError) as e:
        _ = provider_cls(df)
    e.match("Detected dataframe indices of type")
