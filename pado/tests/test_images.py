import pytest

from pado.images import ImageId


@pytest.mark.parametrize(
    "image_id", [
        ImageId("a"),
        ImageId("a", "b", "c"),
        ImageId("a b"),
        ImageId("a\u1234b")
    ]
)
def test_image_id_roundtrip(image_id):
    id_str = image_id.to_str()
    new_id = ImageId.from_str(id_str)
    assert image_id == new_id


def test_image_id(dataset_ro):
    key = next(iter(dataset_ro))
    data = dataset_ro[key]
    assert data["image"].id_str == "i0.tif" == key
