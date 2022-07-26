"""
testing the fsspec file urlpath conversions
"""
from __future__ import annotations

import pytest

from pado.io.files import _OpenFileAndParts
from pado.io.files import find_files
from pado.io.files import urlpathlike_is_localfile
from pado.io.files import urlpathlike_to_string


@pytest.mark.parametrize("str_path", ["gcs://mybucket/image.svs"])
def test_urlpathlike_to_string(str_path):
    assert urlpathlike_to_string(str_path) == "gcs://mybucket/image.svs"


def test_find_files(tmpdir):
    open(tmpdir / "file.svs", "a").close()
    files = find_files(tmpdir, glob="*.svs")

    assert len(files) == 1
    assert isinstance(files[0], _OpenFileAndParts)
    assert files[0].parts[0] == "file.svs"


def test_find_files_empty_urlpath(tmpdir):
    files = find_files(tmpdir)

    assert len(files) == 0


@pytest.mark.parametrize("urlpath", ("file.svs", "non-existent-path"))
def test_find_files_raises_notadirectoryerror(tmpdir, urlpath):
    open(tmpdir / "file.svs", "a").close()
    with pytest.raises(NotADirectoryError):
        find_files(tmpdir / urlpath)


@pytest.mark.parametrize(
    "urlpath, must_exist, expected_result",
    [
        ("file.svs", True, True),
        ("file.svs", False, True),
        ("non-existent-path.svs", False, True),
        ("non-existent-path.svs", True, False),
    ],
)
def test_urlpathlike_is_localfile(tmpdir, urlpath, must_exist, expected_result):
    slide_path = tmpdir.join("file.svs")
    slide_path.write("content")
    assert (
        urlpathlike_is_localfile(tmpdir / urlpath, must_exist=must_exist)
        is expected_result
    )
    assert not urlpathlike_is_localfile(
        "https://example.com/image.svs", must_exist=must_exist
    )
