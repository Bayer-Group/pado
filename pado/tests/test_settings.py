from __future__ import annotations

from pado.settings import pado_config_path


def test_pado_config_path(monkeypatch, tmp_path):
    conf_path = tmp_path.joinpath("mock_config")
    monkeypatch.setenv("PADO_CONFIG_PATH", str(conf_path))
    assert pado_config_path() == conf_path
