"""a place to collect some common settings functionality

data sources can use those to store intermediate data in a common place

A user can override the settings via dynaconf.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf
from dynaconf import ValidationError
from dynaconf import Validator
from platformdirs import user_cache_path
from platformdirs import user_config_path

__all__ = [
    "settings",
    "pado_cache_path",
    "pado_config_path",
]


def __getattr__(name: str) -> Any:
    # compatibility in case someone already relies on this
    if name in {"dataset_registry", "open_registered_dataset"}:
        warnings.warn(
            f"`{name}` has moved to pado.registry", DeprecationWarning, stacklevel=2
        )
        return getattr(__import__("pado.registry"), name)
    raise AttributeError(name)


def validate_registries(value: dict) -> bool:
    if not isinstance(value, dict):
        raise ValidationError(f"registry must be of type dict, got: {value!r}")

    for r_key, r_value in value.items():
        if not isinstance(r_value.urlpath, str):
            raise ValidationError(
                f"registry {r_key} -> urlpath {r_value!r} must be of type str"
            )
    return True


def _as_path(value):
    return os.fspath(Path(value))


def _default_config_path(*_):
    return os.fspath(user_config_path("pado"))


def _default_cache_path(*_):
    return os.fspath(user_cache_path("pado"))


settings = Dynaconf(
    envvar_prefix="PADO",
    settings_file=[".pado.toml"],
    root_path=Path.home(),
    core_loaders=["TOML"],
    validators=[
        Validator("config_path", cast=_as_path, default=_default_config_path),
        Validator("cache_path", cast=_as_path, default=_default_cache_path),
        Validator("override_user_host", must_exist=False, len_min=1),
        Validator("ignore_default_registry", cast=bool, default=False),
        Validator("registry", condition=validate_registries, default={}),
        Validator("allow_pickled_urlpaths", cast=bool, default=False),
        Validator("block_image_id_eval", cast=bool, default=False),
    ],
)


def pado_config_path(pkg: str | None = None, *, ensure_dir: bool = False) -> Path:
    """return the common path for pado config files"""
    pth = Path(settings.config_path)
    if pkg is not None:
        pth = pth.joinpath(pkg)
    if ensure_dir and not pth.is_dir():
        pth.mkdir(parents=True, exist_ok=True)
    return pth


def pado_cache_path(pkg: str | None = None, *, ensure_dir: bool = False) -> Path:
    """return the common path for pado cache files"""
    pth = Path(settings.cache_path)
    if pkg is not None:
        pth = pth.joinpath(pkg)
    if ensure_dir and not pth.is_dir():
        pth.mkdir(parents=True, exist_ok=True)
    return pth
