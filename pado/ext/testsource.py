"""test datasource for pado"""
from contextlib import ExitStack, contextmanager
from functools import cached_property
from itertools import cycle, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tifffile import imsave

from pado.datasource import DataSource, ImageResource

# noinspection PyPep8Naming
from pado.structure import PadoColumn as c

__all__ = ["TestDataSource"]


@contextmanager
def make_temporary_tiff(name, size=(100, 100)):
    """create a temporary tif"""
    with TemporaryDirectory() as tmp_dir:
        data = np.random.randint(0, 255, size=size).astype(np.float32)
        img_fn = Path(tmp_dir).expanduser().absolute() / f"{name}.tif"
        imsave(img_fn, data)
        yield img_fn


class TestImageResource(ImageResource):
    def __init__(self, image_id: str, file: Path):
        self._id = image_id
        self._path = file

    @property
    def id(self) -> Tuple[str, ...]:
        return (self._id,)

    @property
    def path(self) -> Path:
        return self._path


TEST_DATA = {
    # Study
    c.STUDY: ["s0", "s1"],
    # Experiment
    c.EXPERIMENT: ["e0", "e1", "e2"],
    c.EXPERIMENT.subcolumn("DOSE"): ["50ml", "100ml"],
    c.EXPERIMENT.subcolumn("REPEATS"): [1, 2, 3, 4, 5],
    # Group
    c.GROUP: ["g0", "g1", "g3"],
    # Animal
    c.ANIMAL: ["a1", "a2", "a3", "a4", "a5"],
    c.ANIMAL.subcolumn("SPECIES"): ["species_A", "species_B"],
    c.ANIMAL.subcolumn("AGE"): [10, 20, 30, 40],
    # COMPOUND
    c.COMPOUND: ["cA", "cB", "cC"],
    # ORGAN
    c.ORGAN: ["organ_0", "organ_1", "organ_2"],
    # SLIDE
    c.SLIDE: [f"slide_{idx}" for idx in range(10)],
    # IMAGE
    c.IMAGE: ["i0"],
    c.IMAGE.subcolumn("SCANNER"): ["scanner0", "scanner1"],
    # FINDING
    c.FINDING: ["f1", "f2", "f3"],
    c.FINDING.subcolumn("GRADE"): ["minimal", "slight", "moderate", "severe"],
}


def _get_test_data(num_images=3, num_rows=10):
    """generate a test dataset"""
    data = TEST_DATA.copy()
    # set the number of individual images
    data[c.IMAGE] = [f"i{idx}" for idx in range(num_images)]

    # get num_rows items of each list
    for key in data:
        data[key] = list(islice(cycle(data[key]), num_rows))

    return data


class TestDataSource(DataSource):
    identifier = "testsource"
    image_id_columns = [c.IMAGE]

    def __init__(self, num_images=3, num_findings=10):
        self._num_images = num_images
        self._num_findings = num_findings
        self._stack = None
        self._images = []

    def acquire(self, raise_if_missing: bool = True):
        """prepare the temporary test images"""
        _ = raise_if_missing  # ignored
        self._stack = ExitStack()
        for idx in range(self._num_images):
            img = self._stack.enter_context(make_temporary_tiff(f"img_{idx}"))
            self._images.append(img)

    def release(self):
        """release the temporary image resources"""
        if self._stack:
            self._images.clear()
            self._stack.close()
            self._stack = None

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """return the test metadata"""
        if self._stack is None:
            raise RuntimeError("need to access via contextmanager or acquire resource")
        data = _get_test_data(self._num_images, self._num_findings)
        return pd.DataFrame(data)

    def images(self) -> Iterable[ImageResource]:
        """iterate over the test images"""
        if self._stack is None:
            raise RuntimeError("need to access via contextmanager or acquire resource")
        for img_id, img_path in enumerate(self._images):
            yield TestImageResource(f"i{img_id}", img_path)
