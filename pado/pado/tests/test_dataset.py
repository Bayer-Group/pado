import pytest


@pytest.fixture(scope="function")
def pado_dataset(tmp_path):
    pass


def test_opening(tmpdir):
    with pytest.raises():
        ds = PadoDataset(tmpdir)
