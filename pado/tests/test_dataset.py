import shutil
from operator import itemgetter
from pathlib import Path

import pandas as pd
import pytest

from pado.dataset import PadoDataset, is_pado_dataset, verify_pado_dataset_integrity
from pado.datasource import DataSource
from pado.ext.testsource import TestDataSource
from pado.images import SerializableImageResourcesProvider
from pado.metadata import PadoColumn


def count_images(ds: PadoDataset):
    """helper to count images in a pado dataset"""

    def is_image_file(p):
        return p.is_file() and p.name != SerializableImageResourcesProvider.STORAGE_FILE

    images = list(filter(is_image_file, (ds.path / "images").glob("**/*")))
    return len(images)


def test_pado_testsource_verification(datasource: DataSource):
    datasource.verify(acquire=True)


def test_pado_test_datasource_usage(datasource):
    with datasource:
        assert isinstance(datasource.metadata, pd.DataFrame)
        for image in datasource.images:
            assert image.id is not None
            assert image.size > 0
        for annotation in datasource.annotations:
            assert isinstance(annotation, str)


def test_pado_test_datasource_error_without_with(datasource):
    with pytest.raises(RuntimeError):
        _ = datasource.images
    with pytest.raises(RuntimeError):
        _ = datasource.metadata


def test_write_pado_dataset(datasource, tmp_path):
    ds = PadoDataset(path=tmp_path / "my_dataset", mode="x")
    ds.add_source(datasource)
    assert count_images(ds) == 1
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 10


def test_add_multiple_datasets(tmp_path):
    ds = PadoDataset(path=tmp_path / "my_dataset", mode="x")
    ds.add_source(TestDataSource(num_images=2, num_findings=12, identifier="s0"))
    ds.add_source(TestDataSource(num_images=1, num_findings=7, identifier="s1"))
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 19
    assert count_images(ds) == 3


def test_is_pado_dataset(dataset: PadoDataset, tmp_path):
    assert not is_pado_dataset(tmp_path)
    assert is_pado_dataset(dataset.path)


def test_pado_dataset_integrity_fail_dataset(tmp_path):
    with pytest.raises(ValueError):
        verify_pado_dataset_integrity(tmp_path)  # empty folder


def test_pado_dataset_integrity_fail_folders(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path) / "incomplete"
    p.mkdir(parents=True)
    shutil.copytree(dataset.path, p, dirs_exist_ok=True)
    shutil.rmtree(p / "images")  # break the dataset
    with pytest.raises(ValueError):
        verify_pado_dataset_integrity(p)


def test_pado_dataset_integrity_fail_sources(dataset: PadoDataset, tmp_path):
    p = Path(tmp_path) / "incomplete"
    p.mkdir(parents=True)
    shutil.copytree(dataset.path, p, dirs_exist_ok=True)
    for md in p.glob(f"metadata/*"):
        md.unlink(missing_ok=True)  # break the dataset
    with pytest.raises(ValueError):
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


def test_add_source_to_readonly_dataset(dataset_ro, datasource):
    with pytest.raises(RuntimeError):
        dataset_ro.add_source(datasource, copy_images=True)


def test_add_source_twice(datasource, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource, copy_images=True)

    with pytest.raises(ValueError):
        ds.add_source(datasource, copy_images=True)


def test_use_dataset_as_datasource(dataset_ro, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(dataset_ro, copy_images=True)


def test_random_access_dataset(dataset_ro):
    idx = len(dataset_ro) // 2
    md = dataset_ro[idx]
    assert md["image"]
    assert md["metadata"]


def test_iterate_dataset(dataset_ro):
    for md in dataset_ro:
        assert md["image"]
        assert md["metadata"]


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
