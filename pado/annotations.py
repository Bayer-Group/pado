import itertools
import json
import lzma
from abc import ABC
from collections.abc import MutableMapping
from contextlib import suppress
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypedDict,
)

from shapely.geometry import asShape
from shapely.geometry.base import BaseGeometry

from pado.images import ImageId


class AnnotationResource(NamedTuple):
    """keep compatibility with paquo in mind"""

    roi: BaseGeometry
    class_name: Optional[str] = None
    measurements: Optional[List[dict]] = None
    locked: bool = False

    @classmethod
    def from_geojson(cls, geojson: dict) -> "AnnotationResource":
        """create an AnnotationResource from a QuPath geojson dict"""
        cls: "Type['AnnotationsResource']"  # pycharm issue PY-33140
        # note these two asserts are only here for development...
        # these should go...
        # also, this is just relying on the geojson serialization of QuPath
        assert geojson["type"] == "Feature"
        assert geojson["id"] == "PathAnnotationObject"
        properties = geojson["properties"]
        return cls(
            roi=asShape(geojson["geometry"]),
            class_name=properties["classification"].get("name"),
            measurements=properties["measurements"],
            locked=properties["isLocked"],
        )

    @classmethod
    def from_paquo(cls, path_annotation) -> "AnnotationResource":
        """create an AnnotationResource from a paquo QuPathPathAnnotationObject"""
        cls: "Type['AnnotationsResource']"  # pycharm issue PY-33140
        try:
            roi = path_annotation.roi
            path_class = path_annotation.path_class
            class_name = path_class.name if path_class is not None else None
            measurements: List[dict] = path_annotation.measurements.to_records()
            locked: bool = path_annotation.locked
        except AttributeError:
            raise TypeError(
                "path_annotation needs to be a paquo QuPathPathAnnotationObject"
            )
        return cls(
            roi=roi, class_name=class_name, measurements=measurements, locked=locked
        )

    def to_geojson(self) -> dict:
        """return a QuPath compatible geojson representation"""
        return {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "properties": {
                "classification": {"name": self.class_name},
                "isLocked": self.locked,
                "measurements": self.measurements,
            },
        }


class AnnotationsProvider(Mapping[str, Sequence[AnnotationResource]], ABC):
    pass


class AnnotationsDict(TypedDict):
    annotations: List[AnnotationResource]
    metadata: Dict[str, Any]


class SerializableAnnotationsProvider(AnnotationsProvider, MutableMapping):
    STORAGE_FMT = ".geojson.xz"

    def __init__(self, identifier, base_path):
        self._identifier = identifier
        self._path = Path(base_path) / self._identifier
        self._data_paths = {p.stem: p for p in self._path.glob(f"*{self.STORAGE_FMT}")}
        self._data_cache = {}
        self._updated = set()

    def __iter__(self) -> Iterator[ImageId]:
        return iter(set().union(self._data_cache, self._data_paths))

    def __getitem__(self, item: ImageId) -> Sequence[AnnotationResource]:
        try:
            return self._data_cache[item]
        except KeyError:
            fn = self._data_paths[item]
            resource = self._data_cache[item] = self.deserialize_annotations(fn)
            return resource["annotations"]

    def __len__(self) -> int:
        return len(set().union(self._data_cache, self._data_paths))

    def __setitem__(
        self, item: ImageId, annotations_resources: AnnotationsDict
    ) -> None:
        self._data_cache[item] = annotations_resources
        self._updated.add(item)

    def __delitem__(self, item: ImageId) -> None:
        with suppress(KeyError):
            del self._data_cache[item]
        path: Path = self._data_paths.pop(item)
        path.unlink(missing_ok=True)

    ids = Mapping.keys

    def save(self):
        self._path.mkdir(exist_ok=True)
        for image_id in self._updated:
            fn = self._path / f"{image_id}{self.STORAGE_FMT}"
            self.serialize_annotations(fn, self._data_cache[image_id])

    @classmethod
    def from_provider(cls, identifier, base_path, provider):
        inst = cls(identifier, base_path)
        inst.update(provider)
        inst.save()
        return inst

    @classmethod
    def deserialize_annotations(
        cls, file_path: Path, drop_unclassified: bool = True
    ) -> AnnotationsDict:
        # load file
        with lzma.open(file_path, "r") as reader:
            data: dict = json.load(reader)

        # get the annotations
        annotation_dicts = data.pop("annotations", [])
        if drop_unclassified:
            annotation_dicts = [
                a for a in annotation_dicts if "classification" in a["properties"]
            ]

        return AnnotationsDict(
            annotations=list(map(AnnotationResource.from_geojson, annotation_dicts)),
            metadata=data,
        )

    @classmethod
    def serialize_annotations(
        cls, file_path: Path, annotations_dict: AnnotationsDict
    ) -> None:
        data = annotations_dict["metadata"].copy()
        data["annotations"] = [a.to_geojson() for a in annotations_dict["annotations"]]

        with lzma.open(file_path, "w") as writer:
            json.dump(data, writer)


class MergedAnnotationsProvider(AnnotationsProvider):
    def __init__(self, annotation_providers):
        self._providers = list(annotation_providers)
        self._len = len(set().union(*self._providers))
        if self._len < sum(map(len, self._providers)):
            raise ValueError("duplicated keys between providers")

    def __getitem__(self, item: ImageId) -> Sequence[AnnotationResource]:
        for provider in self._providers:
            try:
                return provider[item]
            except KeyError:
                pass
        raise KeyError(item)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[ImageId]:
        return itertools.chain.from_iterable(self._providers)

    def __contains__(self, item) -> bool:
        return any(item in p for p in self._providers)

    def __bool__(self) -> bool:
        return any(self._providers)
