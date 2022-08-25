from __future__ import annotations

import pytest

from pado.annotations import AnnotationProvider
from pado.images.ids import ImageId


@pytest.fixture(scope="function")
def annotations(datasource):
    yield datasource.annotations


def test_datasource_annotation_keys(datasource):
    assert set(datasource.annotations) == {
        ImageId("mock_image_0.svs", site="mock"),
        ImageId("mock_image_1.svs", site="mock"),
        ImageId("mock_image_2.svs", site="mock"),
    }


def test_annotation_serialization_roundtrip(annotations, tmp_path):
    annotations: AnnotationProvider

    p = tmp_path.joinpath("_serialized_provider.annotations.parquet")
    annotations.to_parquet(p)

    new_annotations = AnnotationProvider.from_parquet(p)

    assert annotations is not new_annotations
    assert (
        set(annotations)
        == set(new_annotations)
        == {
            ImageId("mock_image_0.svs", site="mock"),
            ImageId("mock_image_1.svs", site="mock"),
            ImageId("mock_image_2.svs", site="mock"),
        }
    )
    iid = ImageId("mock_image_0.svs", site="mock")

    a0 = annotations[iid][0]
    b0 = new_annotations[iid][0]
    assert a0 == b0
    assert annotations[iid] == new_annotations[iid]
