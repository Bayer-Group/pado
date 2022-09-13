"""some common pado classes"""
from __future__ import annotations

import re
import uuid
from collections import deque
from itertools import repeat
from reprlib import Repr
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import cast
from typing import overload

import pandas as pd

from pado._compat import cached_property
from pado.images.ids import ImageId
from pado.io.store import Store
from pado.types import SerializableItem
from pado.types import UrlpathLike

__all__ = [
    "PadoMutableSequence",
    "PadoMutableSequenceMapping",
    "SerializableProviderMixin",
    "ProviderStoreMixin",
    "GroupedProviderMixin",
    "validate_dataframe_index",
    "is_valid_identifier",
]

_r = Repr()
_r.maxlist = 3

# === collections =============================================================

PI = TypeVar("PI", bound=SerializableItem)


class PadoMutableMapping(MutableMapping[ImageId, PI]):
    __value_type__: Type[PI]

    df: pd.DataFrame
    identifier: str

    def __init__(
        self,
        provider: Mapping[ImageId, PI] | pd.DataFrame | dict | None = None,
        *,
        identifier: Optional[str] = None,
    ):
        if provider is None:
            provider = {}

        if isinstance(provider, type(self)):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            validate_dataframe_index(provider)
            self.df = provider.copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, dict):
            if not provider:
                self.df = pd.DataFrame(columns=self.__value_type__.__fields__)
            else:
                self.df = pd.DataFrame.from_records(
                    index=list(map(ImageId.to_str, provider.keys())),
                    data=list(map(lambda x: x.to_record(), provider.values())),
                    columns=self.__value_type__.__fields__,
                )
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(
                f"expected `{type(self).__name__}`, got: {type(provider).__name__!r}"
            )

    def __getitem__(self, image_id: ImageId) -> PI:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        row = self.df.loc[image_id.to_str()]
        return self.__value_type__.from_obj(row)

    def __setitem__(self, image_id: ImageId, value: PI) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        if not isinstance(value, self.__value_type__):
            raise TypeError(
                f"values must be {self.__value_type__.__name__} instances, got {type(value).__name__!r}"
            )
        dct = value.to_record()
        self.df.loc[image_id.to_str()] = pd.Series(dct)

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        self.df.drop(image_id.to_str(), inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[ImageId]:
        return iter(map(ImageId.from_str, self.df.index))

    def items(self) -> Iterator[tuple[ImageId, PI]]:
        iid_from_str = ImageId.from_str
        value_from_obj = self.__value_type__.from_obj
        for row in self.df.itertuples(index=True, name="ValueAsRow"):
            # noinspection PyProtectedMember
            x = row._asdict()
            i = x.pop("Index")
            yield iid_from_str(i), value_from_obj(x)

    def __repr__(self):
        _akw = [_r.repr_dict(cast(dict, self), 0)]
        if self.identifier is not None:
            _akw.append(f"identifier={self.identifier!r}")
        return f"{type(self).__name__}({', '.join(_akw)})"


PS = TypeVar("PS", bound="PadoMutableSequence")  # todo: use typing.Self


class PadoMutableSequence(MutableSequence[PI]):
    # subclasses must provide this
    __item_class__: Type[PI]

    # annotations for better IDE support
    df: pd.DataFrame

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "__item_class__"):
            raise AttributeError(f"subclass {cls.__name__} must define __item_class__")

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        *,
        image_id: ImageId | None = None,
    ) -> None:
        if df is None:
            self.df = pd.DataFrame(columns=self.__item_class__.__fields__)
        elif isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError(f"requires a pd.DataFrame, not {type(df).__name__}")
        self._image_id = image_id
        if image_id is not None:
            self._update_df_image_id(image_id)

    def __repr__(self):
        v = _r.repr_list(cast(list, self), 0)
        return f"{type(self).__name__}({v}, image_id={self._image_id!r})"

    def __eq__(self, other):
        if not isinstance(other, self.__item_class__):
            return False
        return all(a == b for a, b in zip(self, other))

    @property
    def image_id(self) -> ImageId | None:
        return self._image_id

    def _update_df_image_id(self, image_id: ImageId):
        """internal"""
        if self.df.empty:
            return
        ids = set(self.df["image_id"].unique())
        if len(ids) > 2:
            raise ValueError(f"image_ids in provider not unique: {ids!r}")
        if None not in ids and image_id.to_str() in ids:
            return
        elif {None, image_id.to_str()}.issuperset(ids):
            self.df.loc[self.df["image_id"].isna(), "image_id"] = image_id.to_str()
        else:
            raise AssertionError(
                f"unexpected image_ids in {type(self).__name__}.df: {ids!r}"
            )

    @image_id.setter
    def image_id(self, value: ImageId):
        if not isinstance(value, ImageId):
            raise TypeError(
                f"{value!r} not of type ImageId, got {type(value).__name__}"
            )
        self._update_df_image_id(image_id=value)
        self._image_id = value

    @overload
    def __getitem__(self, index: int) -> PI:
        ...

    @overload
    def __getitem__(self: PS, index: slice) -> PS:
        ...

    def __getitem__(self, index: int | slice) -> PI | PS:
        if isinstance(index, int):
            return self.__item_class__.from_obj(self.df.iloc[index, :])
        elif isinstance(index, slice):
            return self.__class__(self.df.loc[index, :], image_id=self.image_id)
        else:
            raise TypeError(
                f"{type(self).__name__}: indices must be integers or slices, not {type(index).__name__}"
            )

    @overload
    def __setitem__(self, index: int, value: PI) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[PI]) -> None:
        ...

    def __setitem__(self, index: int | slice, value: PI | Iterable[PI]) -> None:
        if isinstance(index, int):
            self.df.iloc[index, :] = pd.DataFrame(
                [value.to_record(self._image_id)],
                columns=self.__item_class__.__fields__,
            )
        elif isinstance(index, slice):
            self.df.iloc[index, :] = pd.DataFrame(
                [x.to_record(self._image_id) for x in value],
                columns=self.__item_class__.__fields__,
            )
        else:
            raise TypeError(
                f"{type(self).__name__}: indices must be integers or slices, not {type(index).__name__}"
            )

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, int):
            self.df.drop(labels=index, axis=0, inplace=True)
        elif isinstance(index, slice):
            self.df.drop(labels=self.df.index[index], axis=0, inplace=True)
        else:
            raise TypeError(
                f"{type(self).__name__}: indices must be integers or slices, not {type(index).__name__}"
            )

    def insert(self, index: int, value: PI) -> None:
        if not isinstance(value, self.__item_class__):
            raise TypeError(
                f"can only insert type {self.__item_class__.__name__}, got {type(value).__name__!r}"
            )
        df_a = self.df.iloc[:index, :]
        df_i = pd.DataFrame(
            [value.to_record(self._image_id)], columns=self.__item_class__.__fields__
        )
        df_b = self.df.iloc[index:, :]
        self.df = pd.concat([df_a, df_i, df_b])

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_records(
        cls: Type[PS],
        annotation_records: Iterable[dict],
        *,
        image_id: ImageId | None = None,
    ) -> PS:
        df = pd.DataFrame(
            list(annotation_records), columns=cls.__item_class__.__fields__
        )
        return cls(df, image_id=image_id)


VT = TypeVar("VT", bound="PadoMutableSequence")


class PadoMutableSequenceMapping(MutableMapping[ImageId, VT]):
    __value_class__: Type[VT]

    df: pd.DataFrame
    identifier: str

    def __init__(
        self,
        provider: Mapping[ImageId, VT] | pd.DataFrame | dict | None = None,
        *,
        identifier: Optional[str] = None,
    ):
        if provider is None:
            provider = {}

        if isinstance(provider, type(self)):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            validate_dataframe_index(provider)
            self.df = provider.copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, dict):
            if not provider:
                self.df = pd.DataFrame(
                    columns=self.__value_class__.__item_class__.__fields__
                )
            else:
                indices = []
                data = []
                for key, value in provider.items():
                    if value is None:
                        continue
                    indices.extend(repeat(ImageId.to_str(key), len(value)))
                    data.extend(a.to_record() for a in value)
                self.df = pd.DataFrame.from_records(
                    index=indices,
                    data=data,
                    columns=self.__value_class__.__item_class__.__fields__,
                )
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(
                f"expected `BaseAnnotationProvider`, got: {type(provider).__name__!r}"
            )

        self._store = {}

    def __getitem__(self, image_id: ImageId) -> VT:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        try:
            return self._store[image_id]
        except KeyError:
            df = self.df.loc[
                [image_id.to_str()], :
            ]  # list: return DataFrame even if length == 1
            df = df.reset_index(drop=True)
            a = self._store[image_id] = self.__value_class__(df, image_id=image_id)
            return a

    def __setitem__(self, image_id: ImageId, v: VT) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        if not isinstance(v, self.__value_class__):
            raise TypeError(f"requires Annotations, got {type(v).__name__}")
        if v.image_id is None:
            v.image_id = image_id
        elif v.image_id != image_id:
            raise ValueError(f"image_ids don't match: {image_id!r} vs {v.image_id!r}")
        self._store[image_id] = v

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        try:
            del self._store[image_id]
        except KeyError:
            had_store = False
        else:
            had_store = True
        try:
            self.df.drop(image_id.to_str(), inplace=True)
        except KeyError:
            had_df = False
        else:
            had_df = True
        if not had_store and not had_df:
            raise KeyError(image_id)

    def __len__(self) -> int:
        if not self._store:
            return self.df.index.nunique()
        else:
            return len(
                set(map(ImageId.from_str, self.df.index.unique())).union(self._store)
            )

    def __iter__(self) -> Iterator[ImageId]:
        return iter(
            set(map(ImageId.from_str, self.df.index.unique())).union(self._store)
        )


PT = TypeVar("PT")
GT = TypeVar("GT", bound="GroupedProviderMixin")


class GroupedProviderMixin:
    __provider_class__: Type[PT]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # todo: should be subclass of mapping with ImageId keys
        # if PadoMutableSequenceMapping not in cls.__mro__:
        #     raise AttributeError(
        #         f"{cls.__name__} must also inherit from a provider class"
        #     )
        if not hasattr(cls, "__provider_class__"):
            raise AttributeError(
                f"subclass {cls.__name__} must define __provider_class__"
            )

    def __init__(self, *providers: PT):
        super().__init__()
        self.providers = []
        for p in providers:
            if not isinstance(p, self.__provider_class__):
                p = self.__provider_class__(p)
            if isinstance(p, type(self)):
                self.providers.extend(p.providers)
            else:
                self.providers.append(p)
        self.__dict__.pop("df")  # clear cache ...

    @cached_property
    def df(self):
        return pd.concat([p.df for p in self.providers])

    def __setitem__(self, image_id: ImageId, value: Any) -> None:
        raise RuntimeError(f"can't add new item to {type(self).__name__}")

    def __delitem__(self, image_id: ImageId) -> None:
        raise RuntimeError(f"can't delete from {type(self).__name__}")

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(map(repr, self.providers))})'

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        # noinspection PyUnresolvedReferences
        super().to_parquet(urlpath, storage_options=storage_options)

    @classmethod
    def from_parquet(cls: Type[GT], urlpath: UrlpathLike) -> GT:
        raise TypeError(f"unsupported operation for {cls.__name__!r}()")


# === mixins ==================================================================

ST = TypeVar("ST", bound=Store)
PC = TypeVar("PC")


class SerializableProviderMixin:
    # required attributes
    __store_class__: Type[ST]

    # these attributes are part of the provider
    df: pd.DataFrame
    identifier: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "__store_class__"):
            raise AttributeError(f"subclass {cls.__name__} must define __store_class__")

    def __repr__(self):
        return f"{type(self).__name__}({self.identifier!r})"

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        store = self.__store_class__()
        store.to_urlpath(
            self.df,
            urlpath,
            identifier=self.identifier,
            storage_options=storage_options,
        )

    @classmethod
    def from_parquet(cls: Type[PC], urlpath: UrlpathLike) -> PC:
        store = cls.__store_class__()
        df, identifier, user_metadata = store.from_urlpath(urlpath)
        if {
            store.METADATA_KEY_STORE_TYPE,
            store.METADATA_KEY_STORE_VERSION,
            store.METADATA_KEY_PADO_VERSION,
            store.METADATA_KEY_PROVIDER_VERSION,
            store.METADATA_KEY_CREATED_AT,
            store.METADATA_KEY_CREATED_BY,
        } != set(user_metadata):
            raise NotImplementedError(f"currently unused {user_metadata!r}")
        inst = cls(identifier=identifier)
        inst.df = df
        return inst


class ProviderStoreMixin(Store):
    """stores the image predictions provider in a single file with metadata"""

    METADATA_KEY_PROVIDER_VERSION: str
    PROVIDER_VERSION: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "METADATA_KEY_PROVIDER_VERSION"):
            raise AttributeError(
                f"subclass {cls.__name__} must define 'METADATA_KEY_PROVIDER_VERSION'"
            )
        if not hasattr(cls, "PROVIDER_VERSION"):
            raise AttributeError(
                f"subclass {cls.__name__} must define 'PROVIDER_VERSION'"
            )

    def __metadata_set_hook__(
        self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]
    ) -> None:
        setter(dct, self.METADATA_KEY_PROVIDER_VERSION, self.PROVIDER_VERSION)
        super().__metadata_set_hook__(dct, setter)

    def __metadata_get_hook__(
        self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]
    ) -> Optional[dict]:
        provider_version = getter(dct, self.METADATA_KEY_PROVIDER_VERSION, None)
        if provider_version is None or provider_version < self.PROVIDER_VERSION:
            raise RuntimeError(
                "Please migrate ImagePredictionsProvider to newer version."
            )
        elif provider_version > self.PROVIDER_VERSION:
            raise RuntimeError(
                "ImageProvider is newer. Please upgrade pado to newer version."
            )
        md = super().__metadata_get_hook__(dct, getter) or {}
        return {
            **md,
            self.METADATA_KEY_PROVIDER_VERSION: provider_version,
        }


# === helpers =================================================================


def validate_dataframe_index(df: pd.DataFrame, *, unique_index: bool = False) -> None:
    """raise if an incorrect index is used"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got: {type(df).__name__!r}")

    try:
        deque(map(ImageId.from_str, df.index), maxlen=0)
    except (TypeError, ValueError):
        idx0 = df.index[0]
        if isinstance(idx0, tuple):
            msg = """\
                Detected dataframe indices of type: tuple
                Did you forget to cast the ImageIds in the index to str?
                >>> df = pd.DataFrame(index=[iid.to_str() for iid in image_ids], data=...)
            """
        else:
            msg = f"""\
                Detected dataframe indices of type: {type(idx0).__name__}
                You have to provide a dataframe with string image ids as an index:
                >>> df = pd.DataFrame(index=[iid.to_str() for iid in image_ids], data=...)
            """
        raise ValueError(dedent(msg))

    if unique_index and not df.index.is_unique:
        raise ValueError("Dataframe index is required to be unique.")


IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9_-]*[a-zA-Z0-9_])?$")


def is_valid_identifier(identifier: str) -> bool:
    """check if an identifier is a valid identifier"""
    if IDENTIFIER_RE.match(identifier):
        return True
    else:
        return False
