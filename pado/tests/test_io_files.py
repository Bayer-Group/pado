"""
testing the fsspec file urlpath conversions
"""
from __future__ import annotations

import pytest

from pado.io.files import urlpathlike_to_string


@pytest.mark.parametrize("str_path", ["gcs://mybucket/image.svs"])
def test_urlpathlike_to_string(str_path):
    assert urlpathlike_to_string(str_path) == "gcs://mybucket/image.svs"
