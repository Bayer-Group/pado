import os
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pado.images import ImageProvider
from pado.images.providers import copy_image
from pado.images.providers import create_image_provider
from pado.images.providers import update_image_provider_urlpaths
from pado.io.files import find_files
from pado.io.paths import match_partial_paths_reversed
from pado.mock import temporary_mock_svs


@pytest.fixture
def multi_image_folder(tmp_path):
    """prepare a folder structure with images"""
    base = tmp_path.joinpath("base_images")
    base.mkdir()
    for idx, subfolder in enumerate(['a_1_x', 'b_2_y', 'c_3_z']):
        s = base.joinpath(subfolder)
        s.mkdir()
        with temporary_mock_svs(f'image_{idx}') as img_fn:
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


def test_match_partial_paths_reversed_does_not_instantiate(multi_image_folder, image_provider):
    # we want this test to fail whenever fsspec.spec.AbstractFileSystem or any of its
    # subclasses gets instantiated. The problem here is, that AbsractFileSystem instances
    # are cached through some metaclass magic in fsspec.spec._Cached.
    # so the way to make sure we fail is to mock __call__ in the metaclass:
    with unittest.mock.patch('fsspec.spec._Cached.__call__', side_effect=RuntimeError):
        m = match_partial_paths_reversed(
            current_urlpaths=image_provider.df.urlpath.values,
            new_urlpaths=list(multi_image_folder.rglob("*.svs")),
        )
        assert len(m) == 3


def test_copy_image(tmp_path, image_provider):
    new_dst = tmp_path.joinpath("new_storage_location")
    iid = next(iter(image_provider))

    copy_image(image_provider, iid, new_dst)

    assert "new_storage_location" in image_provider[iid].urlpath
    assert len(list(find_files(new_dst, glob="**/*.svs"))) == 1
