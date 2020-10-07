import pytest


def test_image_id(dataset_ro):
    data = dataset_ro[0]
    assert data["image"].id_str == "i0.tif"
