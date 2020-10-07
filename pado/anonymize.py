"""provides anonymization for pado Datasets"""
import secrets
from hashlib import sha256
from math import isnan
from numbers import Real

__all__ = ["anonymize", "make_salt"]


def make_salt(num_bytes: int = 32) -> str:
    """generate a salt for anonymization"""
    return secrets.token_hex(2 * num_bytes)


_DEFAULT_SALT = make_salt()


def anonymize(data, *, keep_na=False, mapping=None, salt=None, hasher=sha256) -> str:
    """anonymize data to alpha numeric strings"""
    if salt is None:
        salt = _DEFAULT_SALT

    if keep_na and _is_none_or_nan(data):
        return data

    anonymized = hasher(f"{hash(data)}{salt}".encode()).hexdigest()
    # store map if requested
    if mapping is not None:
        mapping[anonymized] = data

    return anonymized


def _is_none_or_nan(x) -> bool:
    if isinstance(x, Real):
        return isnan(x)
    return x is None
