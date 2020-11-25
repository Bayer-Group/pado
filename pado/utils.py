from collections import ChainMap as _ChainMap
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Mapping, Optional, Set, TypeVar

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class FilteredMapping(Mapping[_KT, _VT]):
    """A FilteredMapping takes a mapping and a set of valid keys to provide
    a readonly view of the underlying mapping with only the intersection of
    valid keys and mapping keys accessible.
    """

    def __init__(self, mapping: Mapping[_KT, _VT], *, valid_keys: Optional[Iterable[_KT]] = None):
        """Initialize a FilteredMapping"""
        if not isinstance(mapping, Mapping):
            raise TypeError("requires a mapping")
        self._mapping = mapping
        self._vk = set(mapping if valid_keys is None else valid_keys)

    @property
    def valid_keys(self) -> Set[_KT]:
        return self._vk

    def __getitem__(self, k: _KT) -> _VT:
        if k not in self.valid_keys:
            raise KeyError(k)
        return self._mapping[k]

    def __len__(self) -> int:
        return len(self.valid_keys.intersection(self._mapping))

    def __iter__(self) -> Iterator[_KT]:
        return iter(self.valid_keys.intersection(self._mapping))


class ChainMap(_ChainMap):
    # this might be a bug in cpython...
    # or maybe not a bug, but at least it's unexpected behavior
    # see: https://bugs.python.org/issue32792
    # after the above patch was applied ChainMap now calls __getitem__ of
    # the underlying maps just to iterate...

    def __iter__(self):
        d = {}  # the original fix uses a dict here basically as an ordered set...
        for mapping in reversed(self.maps):
            # d.update(mapping)  <-- this will call __getitem__ instead of just __iter__ of mapping
            d.update(dict.fromkeys(mapping))
        return iter(d)


def readonly_chain(mappings: Iterable[Mapping[_KT, _VT]]) -> Mapping[_KT, _VT]:
    """merge multiple mappings into one read only mapping"""
    return MappingProxyType(ChainMap(*mappings))


class PriorityChainMap(ChainMap):
    """A PriorityChainMap is a ChainMap that returns the value to a corresponding
    key according to a provided priority_func from the underlying maps
    """
    def __init__(self, *maps: Mapping[_KT, _VT], priority_func: Callable = next):
        super().__init__(*maps)
        self._priority_func = priority_func

    def _iter_maps_getitem_(self, key):
        for mapping in self.maps:
            try:
                yield mapping[key]
            except KeyError:
                pass

    def __getitem__(self, key):
        missing = object()
        value = self._priority_func(self._iter_maps_getitem_(key), missing)
        if value is missing:
            # noinspection PyUnresolvedReferences
            return self.__missing__(key)
        return value


def readonly_priority_chain(
    mappings: Iterable[Mapping[_KT, _VT]],
    priority_func: Callable[[Iterable[_VT]], _VT]
) -> Mapping[_KT, _VT]:
    """merge multiple mappings into one read only priority mapping"""
    return MappingProxyType(PriorityChainMap(*mappings, priority_func=priority_func))