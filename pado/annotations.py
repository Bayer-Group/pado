import json
import lzma
from collections import ChainMap, UserDict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

from shapely.geometry import asShape, mapping
from shapely.geometry.base import BaseGeometry

try:
    from typing import TypedDict  # novermin
except ImportError:
    from typing_extensions import TypedDict


class Annotation(NamedTuple):
    """An Annotation combines geometry, class and additional measurements"""

    roi: BaseGeometry
    class_name: Optional[str] = None
    measurements: Optional[List[dict]] = None
    locked: bool = False  # this is for QuPath consistency and should go


class AnnotationResources(TypedDict):
    """AnnotationResources combine multiple annotations with common metadata"""

    annotations: List[Annotation]
    metadata: Dict[str, Any]


class AnnotationResourcesProvider(UserDict, Mapping[str, AnnotationResources]):
    """An AnnotationResourcesProvider is a map from keys (image_ids) to
    AnnotationResources.

    Lazy loads the annotation resources when requested.
    """

    def __init__(self, path, suffix, load, dump=None):
        """create a new AnnotationResourcesProvider

        Parameters
        ----------
        path :
            the base directory for the serialized annotations per image_id
        suffix :
            the file suffix for each serialized annotation resource
        load :
            function for deserializing AnnotationResources from a file
        dump :
            function for serializing AnnotationResources to a file

        """
        super().__init__()
        path = Path(path)
        path.mkdir(exist_ok=True)
        self._file = lambda key: path.joinpath(key).with_suffix(suffix)
        self._load = load
        self._dump = dump
        self._files = {}
        for p in path.iterdir():
            fn = p.name
            if fn.endswith(suffix):
                self._files[fn[: -len(suffix)]] = p

    def __len__(self):
        return len(set().union(super().__iter__(), self._files))

    def __iter__(self):
        return iter(set().union(super().__iter__(), self._files))

    def __missing__(self, key: str) -> AnnotationResources:
        fn = self._files[key]
        try:
            with fn.open("rt") as fp:
                super()[key] = resources = self._load(fp)
                return resources
        except Exception as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: AnnotationResources):
        if self._dump:
            with self._file(key).open("wb") as fp:
                self._dump(value, fp)
        super().__setitem__(key, value)

    def __delitem__(self, key: str):
        if self._dump:  # file deletion only allowed when dump provided
            if key in self._files:
                fn = self._files.pop(key)
                fn.unlink(missing_ok=True)
        super().__delitem__(key)


def get_provider(path, fmt="geojson") -> AnnotationResourcesProvider:
    """create an AnnotationResourcesProvider for path and format"""
    # todo: determine format from path
    if fmt == "geojson":
        return AnnotationResourcesProvider(
            path, ".geojson.xz", load_geojson, dump_geojson
        )
    else:
        raise ValueError(f"unknown AnnotationResourcesProvider format '{fmt}'")


def store_provider(path, provider, fmt="geojson") -> None:
    """store an AnnotationResourcesProvider at path using the format"""
    store = get_provider(path, fmt)
    store.update(provider)


def merge_providers(providers) -> Mapping[str, AnnotationResources]:
    """merge multiple AnnotationResourceProvider instances into one read only provider"""
    merged = MappingProxyType(ChainMap(*providers))
    if len(merged) < sum(map(len, providers)):
        raise ValueError("duplicated keys between providers")
    return merged


# --- Annotation serialization ------------------------------------------------


def load_geojson(fp, *, drop_unclassified: bool = True) -> AnnotationResources:
    """deserialize an AnnotationResource from a file

    Parameters
    ----------
    fp:
        a file pointer or file path
    drop_unclassified:
        drop an annotation in case it has no class set
    """
    # load file
    with lzma.open(fp, "r") as reader:
        metadata: dict = json.load(reader)

    # get the annotations
    annotation_dicts = metadata.pop("annotations", [])
    if drop_unclassified:
        annotation_dicts = [
            a for a in annotation_dicts if "classification" in a["properties"]
        ]

    annotations = []
    for geojson in annotation_dicts:
        # note these two asserts here should be removed...
        # also, this is just relying on the geojson serialization of QuPath
        assert geojson["type"] == "Feature"
        assert geojson["id"] == "PathAnnotationObject"
        properties = geojson["properties"]
        annotations.append(
            Annotation(
                roi=asShape(geojson["geometry"]),
                class_name=properties["classification"].get("name"),
                measurements=properties["measurements"],
                locked=properties["isLocked"],
            )
        )

    return AnnotationResources(annotations=annotations, metadata=metadata)


def dump_geojson(annotations_dict: AnnotationResources, fp) -> None:
    """serialize an AnnotationResource to a file

    Parameters
    ----------
    annotations_dict:
        a geojson style dictionary of annotations
    fp:
        a file pointer or file path

    """
    data = annotations_dict["metadata"].copy()

    annotations = []
    for a in annotations_dict["annotations"]:
        annotations.append(
            {
                "type": "Feature",
                "id": "PathAnnotationObject",
                "geometry": mapping(a.roi),
                "properties": {
                    "classification": {"name": a.class_name},
                    "isLocked": a.locked,
                    "measurements": a.measurements,
                },
            }
        )

    data["annotations"] = annotations

    # dump file
    with lzma.open(fp, "wt") as writer:
        json.dump(data, writer)
