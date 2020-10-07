import pytest

from pado.annotations import get_provider


@pytest.fixture(scope="function")
def annotations(datasource):
    with datasource:
        yield datasource.annotations


def test_datasource_annotation_keys(datasource):
    with datasource:
        assert set(datasource.annotations) == {"i0.tif"}


def test_annotation_serialization_roundtrip(annotations, tmpdir):
    p = tmpdir.mkdir("annotations")
    provider = get_provider(p)
    provider.update(annotations)

    assert provider is not annotations
    assert set(provider) == set(annotations) == {"i0.tif"}
    assert provider["i0.tif"] == annotations["i0.tif"]
