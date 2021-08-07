import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pado._test_source import make_temporary_tiff
from pado.images import ImageProvider
from pado.images.providers import create_image_provider


@pytest.fixture
def multi_image_folder(tmp_path):
    """prepare a folder structure with images"""
    base = tmp_path.joinpath("base_images")
    base.mkdir()
    for idx, subfolder in enumerate(['a_1_x', 'b_2_y', 'c_3_z']):
        s = base.joinpath(subfolder)
        s.mkdir()
        with make_temporary_tiff(f'image_{idx}') as img_fn:
            data = Path(img_fn).read_bytes()
        s.joinpath(f'image_{idx}.svs').write_bytes(data)
    yield base


@pytest.fixture
def image_provider(multi_image_folder):
    yield create_image_provider(
        search_urlpath=multi_image_folder,
        search_glob="**/*.svs",
        output_urlpath=None,
    )


def test_create_image_provider(multi_image_folder):
    ip = create_image_provider(
        search_urlpath=multi_image_folder,
        search_glob="**/*.svs",
        output_urlpath=None,
    )
    assert len(ip) == 3


def test_write_image_provider(tmp_path, image_provider):
    out = tmp_path.joinpath("images.parquet")
    image_provider.to_parquet(out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_roundtrip_image_provider(image_provider):
    ip0 = image_provider
    with TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, "ip0.parquet")
        ip0.to_parquet(fn)
        ip1 = ImageProvider.from_parquet(fn)

    assert ip0.identifier == ip1.identifier
    assert set(ip0) == set(ip1)
    assert list(ip0.values()) == list(ip1.values())
