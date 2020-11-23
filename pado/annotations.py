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


def check_geometry(geometry, instantiate: bool = True, validate: bool = False) -> bool:
    """
    Returns True iff geometry is a valid shapely/geos geometry.

    From the shapely docs:
    "Shapely does not check the topological simplicity or validity of instances when they
    are constructed as the cost is unwarranted in most cases. Validating factories
    are easily implemented using the :attr:is_valid predicate by users that require them."

    See also:
      https://shapely.readthedocs.io/en/stable/manual.html#object.is_valid
    """
    try:
        if validate:
            return geometry.is_valid
        if instantiate:
            str(geometry)
        return True
    except ValueError:
        return False


class Annotation(NamedTuple):
    """An Annotation combines geometry, class and additional measurements"""

    roi: BaseGeometry
    class_name: Optional[str] = None
    measurements: Optional[List[dict]] = None
    locked: bool = False  # this is for QuPath consistency and should go

    def __repr__(self):
        name = type(self).__name__
        roi = f"'{self.roi.wkt}'"
        cn = f"'{self.class_name or ''}'"
        ms = self.measurements
        if ms:
            return f"{name}(roi={roi}, class_name={cn}, measurements={ms})"
        else:
            return f"{name}(roi={roi}, class_name={cn})"


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
        self._file = lambda key: path.joinpath(key + suffix)
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
        with fn.open("rb") as fp:
            resources = self._load(fp)
        super().__setitem__(key, resources)
        return resources

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

    def __repr__(self):
        return f"{type(self).__name__}({set(self)})"


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
    get_provider(path, fmt).update(provider)


def merge_providers(providers) -> Mapping[str, AnnotationResources]:
    """merge multiple AnnotationResourceProvider instances into one read only provider"""
    merged = MappingProxyType(ChainMap(*providers))
    if len(merged) < sum(map(len, providers)):
        raise ValueError("duplicated keys between providers")
    return merged


# --- Annotation serialization ------------------------------------------------


def load_geojson(fp, *, drop_unclassified: bool = True, drop_broken_geometries: bool = True) -> AnnotationResources:
    """deserialize an AnnotationResource from a file

    Parameters
    ----------
    fp:
        a file pointer or file path
    drop_unclassified:
        drop an annotation in case it has no class set
    """
    # load file
    with lzma.open(fp, "rb") as reader:
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
        shape = asShape(geojson["geometry"])

        if not drop_broken_geometries or check_geometry(shape):
            annotations.append(
                Annotation(
                    roi=shape,
                    class_name=properties["classification"].get("name"),
                    measurements=properties["measurements"],
                    locked=properties["isLocked"],
                )
            )
        else:
            # FIXME Need to add proper logging
            # FIXME meta should carry what is the image_id
            print(f'WARNING: ignored broken geometry for {metadata["project"]} {metadata["scan_name"]}')

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
