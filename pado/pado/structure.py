import string
from abc import abstractmethod
from collections import MutableMapping, MutableSet, Sequence
from operator import attrgetter
from pathlib import Path, PurePath
from typing import Any, Iterator, Union, overload


class Key:
    CHARACTER_SET = string.ascii_letters + string.digits + "_"

    def __init__(self, name: str, parent: "Group"):
        self.is_valid_key(name, raise_if_invalid=True)
        self.name = name
        if not isinstance(parent, Group):
            raise ValueError("requires a parent group")

    def __str__(self):
        return self.name

    @staticmethod
    def is_valid_key(key: Union[str, "Key"], raise_if_invalid=False) -> bool:
        valid, msg = True, ""
        if isinstance(key, Key):
            pass
        elif isinstance(key, str):
            if not set(key).issubset(Key.CHARACTER_SET):
                valid = False, f"unsupported character in key '{key}'"
        else:
            valid = False, f"unsupported type {type(key)}"

        if raise_if_invalid and not valid:
            raise ValueError(msg)
        return valid


class _Group(MutableMapping[str, Union["Group", "File"]]):
    GROUP_SEPERATOR = "/"

    def __setitem__(self, k: _KT, v: _VT) -> None:
        pass

    def __delitem__(self, v: _KT) -> None:
        pass

    def __getitem__(self, k: _KT) -> _VT_co:
        pass

    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[_T_co]:
        pass


class RootGroup(_Group):
    def __init__(self, path: Path):
        self.path = path
        self.parent = None

    def __repr__(self):
        return f"RootGroup('{self.path}')"

    def __str__(self):
        return _Group.GROUP_SEPERATOR

    def __fspath__(self):
        return str(self.path)


def _iter_group_parents(obj):
    while obj is not None:
        yield obj
        obj = obj.parent


class Group(_Group):
    def __init__(self, key: Key, parent: Union["Group", RootGroup]):
        if not isinstance(key, Key):
            raise TypeError("expected a Key instance")
        if not isinstance(parent, _Group):
            raise TypeError("expected a Group or RootGroup")
        self.key = key
        self.parent = parent

    def __repr__(self):
        return f"Group('{self}')"

    def __str__(self):
        keys = [obj.key for obj in _iter_group_parents(self)]
        return self.GROUP_SEPERATOR.join(reversed(keys))

    def __fspath__(self):
        return str(self.path)

    @property
    def path(self):
        objects = reversed(list(_iter_group_parents(self)))
        root = next(objects)
        if not isinstance(root, RootGroup):
            raise RuntimeError("detached group tree")
        path = Path(root.path)
        for obj in objects:
            if not isinstance(obj, Group):
                raise RuntimeError("non-group object in tree")
            path /= obj.key
        return path


class File:
    def __init__(self, key, parent: Union["Group", RootGroup], extension: str):
        self.key = key
        self.parent = parent
        self.extension = extension

    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"

    def __str__(self):
        return f"{str(self.parent)}{_Group.GROUP_SEPERATOR}{str(self.key)}"

    def __fspath__(self):
        return str(self.path)

    @property
    def path(self):
        p = Path(self.parent) / self.key
        return p.with_suffix(self.extension)
