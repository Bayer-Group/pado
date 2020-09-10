"""todo pado anonymizer"""
import hashlib
import math
import numbers
import secrets
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Union


class BaseAnonymizer(ABC):

    _T = TypeVar("_T")

    @abstractmethod
    def __call__(self, item_or_series: _T) -> Union[str, _T]:
        ...


def make_salt(num_bytes=32):
    return secrets.token_hex(2 * num_bytes)


class HashAnonymizer(BaseAnonymizer):
    def __init__(
        self, salt=None, hasher=hashlib.sha256, obscure_missing=False, collect_map=False
    ):
        if salt is None:
            salt = make_salt()
        self._salt = salt
        self._hasher = hasher
        self._obscure_missing = obscure_missing
        # store map if requested
        self.anonymized_map = {} if collect_map else None

    def __call__(self, data: Any) -> Union[str, None, numbers.Real]:
        if not self._obscure_missing:
            if data is None:
                return None
            elif isinstance(data, numbers.Real) and math.isnan(data):
                return data
        else:
            data = str(hash(data))
        anonymized = self._hasher(data + self._salt).hexdigest()
        if self.anonymized_map is not None:
            self.anonymized_map[anonymized] = data
        return anonymized
