from __future__ import annotations

import sys

if sys.version_info >= (3, 8):
    from functools import cached_property as cached_property

else:
    # noinspection PyPep8Naming
    class cached_property:
        _NOCACHE = object()

        def __init__(self, fget):
            self.fget = fget
            self.attrname = None
            self.__doc__ = fget.__doc__

        def __set_name__(self, owner, name):
            self.attrname = name

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self  # pragma: no cover
            cache = obj.__dict__
            val = cache.get(self.attrname, self._NOCACHE)
            if val is self._NOCACHE:
                val = cache[self.attrname] = self.fget(obj)
            return val


__all__ = [
    "cached_property",
]
