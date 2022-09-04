from __future__ import annotations

import pickle
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from pado.annotations import AnnotationProvider
from pado.dataset import DescribeFormat
from pado.dataset import PadoDataset
from pado.dataset import PadoItem
from pado.images.ids import ImageId
from pado.images.image import Image
from pado.images.providers import ImageProvider
from pado.metadata import MetadataProvider
from pado.mock import mock_dataset


def count_images(ds: PadoDataset):
    """helper to count images in a pado dataset"""
    return len(
        [
            f
            for f in ds.filesystem.glob(f"{ds.path}/images/**")
            if ds.filesystem.isfile(f) and not f.endswith(".parquet")
        ]
    )


def test_pado_test_datasource_usage(datasource):
    assert isinstance(datasource.metadata, MetadataProvider)
    for image_id, image in datasource.images.items():
        assert isinstance(image_id, ImageId)
        assert isinstance(image, Image)
        assert image_id is not None
        with image:
            assert image.dimensions.x > 0
            assert image.dimensions.y > 0
    for image_id in datasource.annotations:
        assert isinstance(image_id, ImageId)


def test_pado_test_datasource_image_ids(datasource):
    assert (
        set(datasource.images)
        == set(datasource.metadata)
        == set(datasource.annotations)
        == {
            ImageId("mock_image_0.svs", site="mock"),
            ImageId("mock_image_1.svs", site="mock"),
            ImageId("mock_image_2.svs", site="mock"),
        }
    )


def test_open_dataset(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path)

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        PadoDataset(dataset.urlpath, mode="incorrect_mode")

    with pytest.raises(ValueError):
        PadoDataset(p, mode="r")

    with pytest.raises(FileExistsError):
        PadoDataset(dataset.urlpath, mode="x")

    with pytest.raises(NotADirectoryError):
        PadoDataset(p / "not-found.toml", mode="r")


def test_datasource_image_serializing(datasource, tmp_path):
    ip = ImageProvider(datasource.images)

    for _ in ip.values():
        pass

    assert len(set(ip)) == len(list(ip.values())) == len(ip.df)


def test_image_provider_serializing(datasource, tmp_path):
    ip_old = ImageProvider(datasource.images)
    ip_old.to_parquet(tmp_path / "old.parquet")

    ip = ImageProvider.from_parquet(tmp_path / "old.parquet")
    for _ in ip.values():
        pass

    assert len(set(ip_old)) == len(list(ip.values())) == len(ip.df)
    assert set(ip_old) == set(ip)


def test_reload_dataset(datasource, tmp_path):
    dataset_path = tmp_path / "another_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    assert len(ds.images) == 0
    assert isinstance(ds.images, Mapping)
    _old_ip = ds.images

    # add the source
    ds.ingest_obj(datasource)

    assert ds.images is not _old_ip
    assert len(ds.images) == 3
    assert isinstance(ds.images, Mapping)

    image_ids = set(datasource.images)
    for image_id, image in ds.images.items():
        assert image_id in image_ids


def test_dataset_ro_image_access(dataset_ro):
    im = dataset_ro.images
    list(im)  # get keys
    list(im.values())  # get values


def test_add_source_to_readonly_dataset(dataset_ro, datasource):
    with pytest.raises(RuntimeError):
        dataset_ro.ingest_obj(datasource)


def test_add_source_twice(datasource, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.ingest_obj(datasource)

    with pytest.raises(FileExistsError):
        ds.ingest_obj(datasource)


def test_datasource_df(datasource):
    assert len(datasource.images) > 0
    for img_id, img in datasource.images.items():
        assert isinstance(img_id, ImageId)
        assert isinstance(img, Image)


def test_dataset_ro_df_len(dataset_ro):
    assert len(dataset_ro.images) > 0
    if isinstance(dataset_ro, ImageProvider):
        assert len(dataset_ro.images._df) == len(dataset_ro.images)


def test_random_access_dataset(dataset_ro):
    image_ids = list(dataset_ro.images)
    idx = len(image_ids) // 2
    md = dataset_ro.metadata[image_ids[idx]]
    assert isinstance(md, pd.DataFrame)
    assert md.size > 0


def test_ingested_dataset_attribute_retrieval(datasource):
    ds = PadoDataset(None, mode="x")
    ds.ingest_obj(datasource)

    image_id = next(iter(ds.images))
    _ = ds.annotations[image_id]
    _ = ds.metadata[image_id]
    _ = ds.images[image_id]


def test_dataset_caches_after_ingest(dataset_ro):
    ds = PadoDataset(None, mode="w")
    ds.ingest_obj(dataset_ro)

    assert not ds.images.df.empty
    assert not ds.metadata.df.empty
    assert not ds.annotations.df.empty

    ds.ingest_obj(ImageProvider(dataset_ro.images, identifier="something"))
    assert not ds.images.df.empty
    ds.ingest_obj(MetadataProvider(dataset_ro.metadata, identifier="something"))
    assert not ds.metadata.df.empty
    ds.ingest_obj(AnnotationProvider(dataset_ro.annotations, identifier="something"))
    assert not ds.annotations.df.empty


@pytest.mark.parametrize("idx", list(range(3)))
def test_dataset_getitem_padoitem(dataset, idx):
    pado_item_idx = dataset[idx]
    image_id = dataset.index[idx]
    pado_item_image_id = dataset[image_id]

    assert isinstance(pado_item_idx, PadoItem)
    assert isinstance(pado_item_image_id, PadoItem)
    assert pado_item_idx.id == pado_item_image_id.id
    assert pado_item_idx.image == pado_item_image_id.image
    assert pado_item_idx.annotations == pado_item_image_id.annotations
    pd.testing.assert_frame_equal(pado_item_idx.metadata, pado_item_image_id.metadata)


def test_dataset_getitem_slice(dataset):
    pad_dataset = dataset[0:2]
    assert isinstance(pad_dataset, PadoDataset)
    assert len(pad_dataset.index) == 2
    for pado_item in pad_dataset:
        assert isinstance(pado_item, PadoItem)


@pytest.mark.parametrize("arg", (None, {}, 9.1, "something"))
def test_dataset_getitem_raises_typeerror(dataset, arg):
    with pytest.raises(TypeError):
        dataset[arg]


@pytest.mark.parametrize("arg", (None, {}, 9.1, "something", 0))
def test_dataset_get_by_id_raises_typeerror(dataset, arg):
    with pytest.raises(TypeError):
        dataset.get_by_id(arg)


@pytest.mark.parametrize("arg", (None, {}, 9.1, "something", ImageId("hello")))
def test_dataset_get_by_idx_raises_typeerror(dataset, arg):
    with pytest.raises(TypeError):
        dataset.get_by_idx(arg)


DESCRIPTION_KEYS = {
    "avg_annotations_per_image",
    "avg_image_height",
    "avg_image_size",
    "avg_image_width",
    "common_classes_area",
    "common_classes_count",
    "metadata_columns",
    "num_images",
    "num_mpps",
    "path",
    "total_num_annotations",
    "total_size_images",
}


def test_dataset_describe(dataset):
    output = dataset.describe(output_format=DescribeFormat.JSON)
    assert set(output) == DESCRIPTION_KEYS
    assert output["num_images"] == 3


def test_dataset_describe_ip_only(dataset_ip_only):
    output = dataset_ip_only.describe(output_format=DescribeFormat.JSON)
    assert output["metadata_columns"] == []
    assert output["avg_annotations_per_image"]["val"] == 0


def test_dataset_pickle_fsspec_memory_dataset():
    ds = mock_dataset(None)  # create an in memory dataset

    with pytest.warns(UserWarning, match=r".*memory"):
        _ = pickle.dumps(ds)


def test_dataset_unpickle_fsspec_memory_dataset_same_process():
    ds = mock_dataset(None)  # create an in memory dataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickled = pickle.dumps(ds)

    with pytest.warns(UserWarning, match=r"Key collision .*memory"):
        ds = pickle.loads(pickled)  # nosec B301

    assert ds[0].image is not None  # test if we can access the ds


def test_dataset_unpickle_fsspec_memory_dataset_different_process(tmp_path):
    ds = mock_dataset(None)  # create an in memory dataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickled = pickle.dumps(ds)

    load_pickle_py = tmp_path.joinpath("load_pickle.py")
    load_pickle_py.write_text(
        textwrap.dedent(
            f"""\
            import pickle
            import warnings

            pickled = {pickled!r}

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ds = pickle.loads(pickled)
            assert ds[0].image is not None  # test if we can access the ds

            """
        )
    )
    out = subprocess.run([sys.executable, load_pickle_py], capture_output=True)
    assert out.returncode == 0, out.stderr.decode()


def test_empty_dataset():
    ds = PadoDataset(None)
    assert len(ds.images) == 0
    assert len(ds.metadata) == 0
    assert len(ds.annotations) == 0


def test_dataset_filter_empty(dataset_ro):
    ds = dataset_ro.filter([], on_empty="ignore")
    assert len(ds.images) == 0
    assert len(ds.metadata) == 0
    assert len(ds.annotations) == 0
    assert (ds.metadata.df.columns == dataset_ro.metadata.df.columns).all()


def test_open_dataset_frozen(mock_dataset_path: str):
    Path(mock_dataset_path).joinpath(".frozen").touch()

    with pytest.raises(PermissionError):
        _ = PadoDataset(mock_dataset_path, mode="r+")
