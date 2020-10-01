"""test datasource for pado"""
import hashlib
import random
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pado.annotations import AnnotationResourcesProvider
from pado.datasource import DataSource
from pado.fileutils import hash_file
from pado.images import ImageResource, ImageResourcesProvider, LocalImageResource
from pado.metadata import PadoColumn

try:
    from pado._version import __version__
except ImportError:
    __version__ = "not-installed"

try:
    from tifffile import imsave
except ImportError:  # pragma: no cover
    raise ImportError("pado.ext.testsource requires the `pado[testsource]` extra")

__all__ = ["TestDataSource"]


@contextmanager
def make_temporary_tiff(name, size=(100, 100)):
    """create a temporary tif"""
    with TemporaryDirectory() as tmp_dir:
        data = np.random.randint(0, 255, size=size).astype(np.float32)
        img_fn = Path(tmp_dir).expanduser().absolute() / f"{name}.tif"
        imsave(img_fn, data)
        yield img_fn


def _get_test_data(num_images=3, num_rows=10):
    """generate a test dataset"""
    c = PadoColumn
    records = []

    # findings per image
    num_findings_per_image = num_rows // num_images
    num_findings = [num_findings_per_image] * num_images
    num_findings[-1] += num_rows - (num_findings_per_image * num_images)

    # set the number of individual images
    for idx, num_f in zip(range(num_images), num_findings):
        image = f"i{idx}.tif"
        image_scanner = random.choice(["scanner0", "scanner1"])
        slide = f"slide_{idx}"
        organ = f"o{idx}"
        animal = f"a{idx}"
        animal_species = random.choice(["species_A", "species_B"])
        animal_age = random.randint(1, 4) * 10
        group = f"g{idx}"
        exp = f"e{idx}"
        exp_dose = random.choice(["50ml", "100ml"])
        exp_repeats = random.randint(1, 6)
        compound = f"c{idx}"
        study = "s0"
        for finding in range(num_f):
            finding = f"f{idx}_{finding}"
            finding_grade = random.choice(["minimal", "slight", "moderate", "severe"])

            record = {
                c.STUDY: study,
                c.EXPERIMENT: exp,
                c.EXPERIMENT.subcolumn("DOSE"): exp_dose,
                c.EXPERIMENT.subcolumn("REPEATS"): exp_repeats,
                c.GROUP: group,
                c.ANIMAL: animal,
                c.ANIMAL.subcolumn("SPECIES"): animal_species,
                c.ANIMAL.subcolumn("AGE"): animal_age,
                c.COMPOUND: compound,
                c.ORGAN: organ,
                c.SLIDE: slide,
                c.IMAGE: image,
                c.IMAGE.subcolumn("SCANNER"): image_scanner,
                c.FINDING: finding,
                c.FINDING.subcolumn("GRADE"): finding_grade,
            }

            records.append(record)

    return records


class _TestImageResourcesProvider(ImageResourcesProvider):
    def __init__(self, images):
        self._images = images

    def __getitem__(self, item: int) -> ImageResource:
        img_id, img_path, img_md5 = self._images[item]
        return LocalImageResource(img_id, img_path, img_md5)

    def __len__(self) -> int:
        return len(self._images)


class TestDataSource(DataSource):
    identifier = "testsource"

    def __init__(self, num_images=3, num_findings=10, identifier=None):
        self._num_images = num_images
        self._num_findings = num_findings
        self._stack = None
        self._images = []
        if identifier is not None:
            self.identifier = identifier  # allow overriding identifier
        # cache
        self._md = None
        self._im = None

    def acquire(self, raise_if_missing: bool = True):
        """prepare the temporary test images"""
        _ = raise_if_missing  # ignored
        if self._stack:
            return
        self._stack = ExitStack()
        for idx in range(self._num_images):
            img_id = f"i{idx}.tif"
            img = self._stack.enter_context(make_temporary_tiff(f"img_{idx}"))
            md5 = hash_file(img, hasher=hashlib.md5)
            self._images.append((img_id, img, md5))

    def release(self):
        """release the temporary image resources"""
        if self._stack:
            self._images.clear()
            self._stack.close()
            self._stack = None

    @property
    def metadata(self) -> pd.DataFrame:
        """return the test metadata"""
        if self._stack is None:
            raise RuntimeError("need to access via contextmanager or acquire resource")
        if self._md is None:
            data = _get_test_data(self._num_images, self._num_findings)
            self._md = pd.DataFrame.from_records(data)
        return self._md

    @property
    def images(self) -> ImageResourcesProvider:
        """iterate over the test images"""
        if self._stack is None:
            raise RuntimeError("need to access via contextmanager or acquire resource")
        if self._im is None:
            self._im = _TestImageResourcesProvider(self._images)
        return self._im

    @property
    def annotations(self) -> AnnotationResourcesProvider:
        # noinspection PyTypeChecker
        return {}
