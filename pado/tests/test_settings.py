from __future__ import annotations

from pado.settings import pado_config_path
from pado.settings import settings


def test_pado_config_path(tmp_path):
    conf_path = tmp_path.joinpath("mock_config")
    settings.configure(config_path=str(conf_path))
    assert pado_config_path() == conf_path
