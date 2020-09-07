import pandas as pd


def test_pado_test_datasource(datasource):
    assert isinstance(datasource.metadata, pd.DataFrame)
    for image in datasource.images():
        assert image.id is not None
        assert image.path.is_file()
