import glob
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from pado.dataset import PadoDataset
from pado.images import ImageId
from pado.images import ImageProvider
from pado.images import Image
from pado.metadata import MetadataProvider


def count_images(ds: PadoDataset):
    """helper to count images in a pado dataset"""
    return len([
        f for f in ds.filesystem.glob(f"{ds.path}/images/**")
        if ds.filesystem.isfile(f) and not f.endswith(".parquet")
    ])


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
    assert set(
        datasource.images
    ) == set(
        datasource.metadata
    ) == set(
        datasource.annotations
    ) == {
        ImageId('mock_image_0.svs', site='mock'),
        ImageId('mock_image_1.svs', site='mock'),
        ImageId('mock_image_2.svs', site='mock'),
    }


@pytest.mark.xfail
def test_open_dataset(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path)

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        PadoDataset(dataset.urlpath, mode="incorrect_mode")

    with pytest.raises(FileExistsError):
        PadoDataset(dataset.urlpath, mode="x")

    with pytest.raises(FileNotFoundError):
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


@pytest.mark.xfail
def test_add_source_twice(datasource, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.ingest_obj(datasource)

    with pytest.raises(ValueError):
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


@pytest.mark.xfail
def test_iterate_dataset(dataset_ro):
    for k in dataset_ro:
        md = dataset_ro[k]
        assert {"image", "metadata", "annotations"}.issubset(md)
