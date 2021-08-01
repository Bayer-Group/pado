import json
import lzma
import os
import warnings
from collections import UserDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional

import fsspec
from shapely.geometry import asShape, mapping
from shapely.geometry.base import BaseGeometry

from pado.image import ImageId

try:
    from typing import TypedDict  # novermin
except ImportError:
    from typing_extensions import TypedDict


def check_geometry(geometry, *, instantiate: bool = True, validate: bool = False) -> bool:
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


class AnnotationResourcesProvider(UserDict, Mapping[ImageId, AnnotationResources]):
    """An AnnotationResourcesProvider is a map from keys (image_ids) to
    AnnotationResources.

    Lazy loads the annotation resources when requested.
    """

    LEGACY_SUPPORT_SEPARATOR = "__"

    def __init__(self, fs, fspath, suffix, load, dump=None, legacy_support=True):
        """create a new AnnotationResourcesProvider

        Parameters
        ----------
        fspath :
            the base directory for the serialized annotations per image_id
        suffix :
            the file suffix for each serialized annotation resource
        load :
            function for deserializing AnnotationResources from a file
        dump :
            function for serializing AnnotationResources to a file

        """
        super().__init__()

        fs.mkdirs(fspath, exist_ok=True)

        self._path = fspath
        self._fs = fs
        self._suffix = suffix
        self._load = load
        self._dump = dump
        self._files: Dict[ImageId, str] = {}

        legacy_recovered = 0
        for current, dirs, files in fs.walk(fspath, maxdepth=1):
            for fn in files:
                if not fn.endswith(suffix):
                    continue  # skip if not an annotation file

                fn = fn[: -len(suffix)]
                try:
                    image_id = ImageId.from_str(fn)
                except ValueError:
                    if not legacy_support:
                        continue
                    image_id = ImageId(*fn.split(self.LEGACY_SUPPORT_SEPARATOR))
                    legacy_recovered += 1

                self._files[image_id] = os.path.join(current, fn + self._suffix)

        if legacy_recovered > 0:
            warnings.warn(f"legacy: converted {legacy_recovered} annotations to newer annotation storage fmt")

    def _file(self, key: ImageId) -> str:
        assert isinstance(key, ImageId)
        p = os.path.join(self._path, f"{key.to_str()}{self._suffix}")
        return p

    def __len__(self):
        return len(set().union(super().__iter__(), self._files))

    def __iter__(self) -> Iterator[ImageId]:
        return iter(set().union(super().__iter__(), self._files))

    def __contains__(self, key: ImageId) -> bool:
        return key in self.data or key in self._files

    def __missing__(self, key: ImageId) -> AnnotationResources:
        fn = self._files[key]
        try:
            with self._fs.open(fn, mode="rb") as fp:
                resources = self._load(fp)
        except FileNotFoundError:
            raise KeyError(key)
        super().__setitem__(key, resources)
        return resources

    def __setitem__(self, key: ImageId, value: AnnotationResources):
        if self._dump:
            with self._fs.open(self._file(key), mode="wb") as fp:
                self._dump(value, fp)
        super().__setitem__(key, value)

    def __delitem__(self, key: ImageId):
        # FIXME
        if self._dump:  # file deletion only allowed when dump provided
            if key in self._files:
                fn = self._files.pop(key)
                try:
                    self._fs.rm(fn)
                except FileNotFoundError:
                    pass
        super().__delitem__(key)

    def __repr__(self):
        return f"{type(self).__name__}({set(self)})"


def get_provider(fs, path, fmt="geojson") -> AnnotationResourcesProvider:
    """create an AnnotationResourcesProvider for path and format"""
    # todo: determine format from path
    if fmt == "geojson":
        return AnnotationResourcesProvider(
            fs, path, ".geojson.xz", load_geojson, dump_geojson
        )
    else:
        raise ValueError(f"unknown AnnotationResourcesProvider format '{fmt}'")


def store_provider(fs, path, provider, fmt="geojson") -> None:
    """store an AnnotationResourcesProvider at path using the format"""
    get_provider(fs, path, fmt).update(provider)


# --- Annotation serialization ------------------------------------------------


def load_geojson(fp, *, drop_unclassified: bool = True, drop_broken_geometries: bool = True) -> AnnotationResources:
    """deserialize an AnnotationResource from a file

    Parameters
    ----------
    fp:
        a file pointer or file path
    drop_unclassified:
        drop an annotation in case it has no class set
    drop_broken_geometries:
        drop an annotation in case it is broken

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
