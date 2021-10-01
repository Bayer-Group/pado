"""a place to collect some common settings functionality

data sources can use those to store intermediate data in a common place

A user can override the settings via dynaconf.
"""
from __future__ import annotations

from pathlib import Path

from dynaconf import Dynaconf
from dynaconf import Validator
from platformdirs import user_cache_path
from platformdirs import user_config_path

__all__ = [
    'pado_cache_path',
    'pado_config_path',
]

settings = Dynaconf(
    envvar_prefix="PADO",
    settings_file=[".pado.toml"],
    root_path=Path.home(),
    core_loaders=['TOML'],
    validators=[
        Validator('config_path', cast=Path, default=user_config_path('pado')),
        Validator('cache_path', cast=Path, default=user_cache_path('pado')),
    ],
)


def pado_config_path(pkg: str | None = None, *, ensure_dir: bool = False) -> Path:
    """return the common path for pado config files"""
    pth = settings.config_path
    if pkg is not None:
        pth = pth.joinpath(pkg)
    if ensure_dir and not pth.is_dir():
        pth.mkdir(parents=True, exist_ok=True)
    return pth


def pado_cache_path(pkg: str | None = None, *, ensure_dir: bool = False) -> Path:
    """return the common path for pado cache files"""
    pth = settings.cache_path
    if pkg is not None:
        pth = pth.joinpath(pkg)
    if ensure_dir and not pth.is_dir():
        pth.mkdir(parents=True, exist_ok=True)
    return pth