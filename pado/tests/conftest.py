import hashlib
import pathlib
import shutil
import urllib.request
from typing import Iterable

import pandas as pd
import pytest

from pado.datasource import DataSource, ImageResource
from pado.structure import PadoColumn

# openslide aperio test images
IMAGES_BASE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"


def md5(fn):
    m = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


@pytest.fixture(scope="session")
def svs_small():
    """download the smallest aperio test image svs"""
    small_image = "CMU-1-Small-Region.svs"
    small_image_md5 = "1ad6e35c9d17e4d85fb7e3143b328efe"
    data_dir = pathlib.Path(__file__).parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    img_fn = data_dir / small_image

    if not img_fn.is_file():
        # download svs from openslide test images
        url = IMAGES_BASE_URL + small_image
        with urllib.request.urlopen(url) as response, open(img_fn, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

    if md5(img_fn) != small_image_md5:  # pragma: no cover
        shutil.rmtree(img_fn)
        pytest.fail("incorrect md5")
    else:
        yield img_fn.absolute()


class TestImageResource(ImageResource):
    def __init__(self, test_id, test_file_path):
        self._id = test_id
        self._path = test_file_path

    @property
    def id(self):
        return self._id

    @property
    def path(self):
        return self._path


class TestDataSource(DataSource):
    identifier = "testsource"
    image_id_columns = [PadoColumn.COMPOUND, PadoColumn.IMAGE]

    def __init__(self, test_image_path):
        self._image_path = test_image_path

    @property
    def metadata(self) -> pd.DataFrame:
        data = {c: range(3) for c in PadoColumn}
        data[PadoColumn.COMPOUND] = ["abc", "abc", "efg"]
        data[PadoColumn.IMAGE] = ["a.svs", "b.svs", "c.svs"]
        return pd.DataFrame(data)

    def images(self) -> Iterable[ImageResource]:
        for _, image_id in self.metadata[self.image_id_columns].iterrows():
            yield TestImageResource(tuple(image_id.tolist()), self._image_path)


@pytest.fixture(scope="function")
def datasource(svs_small):
    yield TestDataSource(svs_small)
