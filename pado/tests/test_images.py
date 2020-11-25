import pytest


def test_image_id(dataset_ro):
    key = next(iter(dataset_ro))
    data = dataset_ro[key]
    assert data["image"].id_str == "i0.tif" == key
