import os

import fsspec
import pytest

from pado.annotations import get_provider
from pado.images import ImageId


@pytest.fixture(scope="function")
def annotations(datasource):
    with datasource:
        yield datasource.annotations


def test_datasource_annotation_keys(datasource):
    with datasource:
        assert set(datasource.annotations) == {ImageId("i0.tif")}


def test_annotation_serialization_roundtrip(annotations, tmpdir):
    p = tmpdir.mkdir("annotations")
    fs, _, [path] = fsspec.get_fs_token_paths(os.fspath(p))
    provider = get_provider(fs, path)
    provider.update(annotations)

    assert provider is not annotations
    assert set(provider) == set(annotations) == {ImageId("i0.tif")}
    assert provider[ImageId("i0.tif")] == annotations[ImageId("i0.tif")]
