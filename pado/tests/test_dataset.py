import glob
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from pado._test_source import TestDataSource
from pado.dataset import PadoDataset, is_pado_dataset, verify_pado_dataset_integrity
from pado.datasource import DataSource
from pado.images import ImageId, ImageProvider
from pado.img import Image
from pado.metadata import PadoColumn


def count_images(ds: PadoDataset):
    """helper to count images in a pado dataset"""
    return len([
        f for f in ds.filesystem.glob(f"{ds.path}/images/**")
        if ds.filesystem.isfile(f) and not f.endswith(".parquet")
    ])


def test_pado_testsource_verification(datasource: DataSource):
    from pado.datasource import verify_datasource

    assert verify_datasource(datasource, acquire=True)


def test_pado_test_datasource_usage(datasource):
    with datasource:
        assert isinstance(datasource.metadata, pd.DataFrame)
        for image_id, image in datasource.images.items():
            assert isinstance(image_id, ImageId)
            assert isinstance(image, Image)
            assert image_id is not None
            with image:
                assert image.get_size() > (0, 0)
        for image_id in datasource.annotations:
            assert isinstance(image_id, ImageId)


def test_pado_test_datasource_error_without_with(datasource):
    with pytest.raises(RuntimeError):
        _ = datasource.images
    with pytest.raises(RuntimeError):
        _ = datasource.metadata


def test_pado_test_datasource_image_ids(datasource):
    # TODO: revisit
    datasource.acquire()
    assert set(datasource.images) == {ImageId("i0.tif")}
    assert set(map(ImageId.from_str, datasource.metadata[PadoColumn.IMAGE])) == {ImageId("i0.tif")}
    assert set(datasource.annotations) == {ImageId("i0.tif")}


def test_write_pado_dataset(datasource, tmp_path):
    ds = PadoDataset(urlpath=tmp_path / "my_dataset", mode="x")
    ds.add_source(datasource)
    assert count_images(ds) == 1
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 10


def test_add_multiple_datasets(tmp_path):
    ds = PadoDataset(urlpath=tmp_path / "my_dataset", mode="x")
    ds.add_source(TestDataSource(num_images=2, num_findings=12, identifier="s0"))
    ds.add_source(TestDataSource(num_images=1, num_findings=7, identifier="s1"))
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 19
    assert count_images(ds) == 3


def test_is_pado_dataset(dataset: PadoDataset, tmp_path):
    assert not is_pado_dataset(tmp_path)
    assert is_pado_dataset(dataset.path)


def test_pado_dataset_integrity_fail_dataset(tmp_path):
    with pytest.raises(ValueError, match=".* not a pado dataset"):
        verify_pado_dataset_integrity(tmp_path)  # empty folder


def test_pado_dataset_integrity_fail_folders(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path) / "incomplete"
    shutil.copytree(dataset.path, p)
    shutil.rmtree(p / "images")  # break the dataset
    with pytest.raises(ValueError, match="missing .* directory"):
        verify_pado_dataset_integrity(p)


def test_pado_dataset_integrity_fail_sources(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path) / "incomplete"
    shutil.copytree(dataset.path, p)
    for md in glob.glob(os.fspath(p / f"metadata/*")):
        os.unlink(md)  # break the dataset
    with pytest.raises(ValueError, match=".* missing metadata"):
        verify_pado_dataset_integrity(p)


def test_open_dataset(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path)

    with pytest.raises(ValueError):
        PadoDataset(p / "wrong_suffix.abc")

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        PadoDataset(dataset.path, mode="incorrect_mode")

    with pytest.raises(FileNotFoundError):
        PadoDataset(p / "not-found.toml", mode="r")

    with pytest.raises(FileExistsError):
        PadoDataset(dataset.path, mode="x")


def test_open_dataset_with_integrity_errors(dataset: PadoDataset):
    shutil.rmtree(dataset.path / "images")
    with pytest.raises(RuntimeError):
        PadoDataset(dataset.path, mode="r")


def test_open_pado_version_too_old(dataset: PadoDataset, monkeypatch):
    # pretend pado is older
    monkeypatch.setattr(PadoDataset, "__version__", PadoDataset.__version__ - 1)
    with pytest.raises(RuntimeError):
        PadoDataset(dataset.path)


def test_pado_dataset_open_with_different_identifier(dataset: PadoDataset):
    ds = PadoDataset(dataset.path, mode="r+", identifier="new_identifier")
    assert ds.identifier == "new_identifier"


def test_datasource_image_serializing(datasource, tmp_path):
    with datasource:
        ip = ImageProvider(datasource.images)

    for _ in ip.values():
        pass

    assert len(set(ip)) == len(list(ip.values())) == len(ip.df)


def test_image_provider_serializing(datasource, tmp_path):
    with datasource:
        ip_old = ImageProvider(datasource.images)
        ip_old.to_parquet(tmp_path / "old.parquet")

    ip = ImageProvider.from_parquet(tmp_path / "old.parquet")
    for _ in ip.values():
        pass

    assert len(set(ip_old)) == len(list(ip.values())) == len(ip.df)
    assert set(ip_old) == set(ip)


@pytest.mark.parametrize(
    "copy_images", [False, True], ids=["nocopy", "copy"]
)
def test_reload_dataset(datasource, tmp_path, copy_images):
    dataset_path = tmp_path / "another_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    assert len(ds.images) == 0
    assert isinstance(ds.images, Mapping)
    _old_ip = ds.images

    # add the source
    ds.add_source(datasource, copy_images=copy_images)

    assert ds.images is not _old_ip
    assert len(ds.images) == 1
    assert isinstance(ds.images, Mapping)

    with datasource:
        image_ids = set(datasource.images)
    for image_id, image in ds.images.items():
        assert image_id in image_ids


def test_dataset_ro_image_access(dataset_ro):
    im = dataset_ro.images
    list(im)  # get keys
    list(im.values())  # get values


def test_add_source_to_readonly_dataset(dataset_ro, datasource):
    with pytest.raises(RuntimeError):
        dataset_ro.add_source(datasource, copy_images=True)


def test_add_source_twice(datasource, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource, copy_images=True)

    with pytest.raises(ValueError):
        ds.add_source(datasource, copy_images=True)


def test_datasource_df(datasource):
    with datasource:
        assert len(datasource.images) > 0
        for img_id, img in datasource.images.items():
            assert isinstance(img_id, ImageId)
            assert isinstance(img, Image)


def test_dataset_ro_df_len(dataset_ro):
    assert len(dataset_ro.images) > 0
    if isinstance(dataset_ro, ImageProvider):
        assert len(dataset_ro.images._df) == len(dataset_ro.images)


def test_use_dataset_as_datasource(dataset_ro, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    assert set(dataset_ro.annotations) == {ImageId("i0.tif")}
    ds.add_source(dataset_ro, copy_images=True)


def test_random_access_dataset(dataset_ro):
    image_ids = list(dataset_ro)
    idx = len(image_ids) // 2
    md = dataset_ro[image_ids[idx]]
    assert md["image"]
    assert isinstance(md["metadata"], pd.DataFrame)
    assert md["metadata"].size > 0
    assert "annotations" in md


def test_iterate_dataset(dataset_ro):
    for k in dataset_ro:
        md = dataset_ro[k]
        assert {"image", "metadata", "annotations"}.issubset(md)


@pytest.mark.parametrize(
    "accessor,column",
    [
        ("studies", PadoColumn.STUDY),
        ("experiments", PadoColumn.EXPERIMENT),
        ("groups", PadoColumn.GROUP),
        ("animals", PadoColumn.ANIMAL),
        ("compounds", PadoColumn.COMPOUND),
        ("organs", PadoColumn.ORGAN),
        ("slides", PadoColumn.SLIDE),
        ("images", PadoColumn.IMAGE),
        ("findings", PadoColumn.FINDING),
    ],
)
def test_pado_dataframe_accessor(dataset, accessor, column):
    # this is testing, i.e.: dataset.metadata.pado.organs
    df_subset = getattr(dataset.metadata.pado, accessor)
    assert isinstance(df_subset, pd.DataFrame)
    assert len(df_subset) > 0
    assert all(map(lambda x: x.startswith(column), df_subset.columns))


def test_pado_dfa_with_incorrect_df(dataset):
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(AttributeError):
        _ = df.pado

    df = dataset.metadata
    # "-" is an invalid column name character
    df["IMAGE__ABC-INCORRECT"] = df["IMAGE"]
    with pytest.raises(AttributeError):
        _ = df.pado
