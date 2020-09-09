import pytest

from pado.ext.testsource import TestDataSource


@pytest.fixture(scope="function")
def datasource():
    yield TestDataSource(num_images=1, num_findings=10)
