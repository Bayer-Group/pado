"""different Annotation formats"""
from __future__ import annotations

import enum
import struct
from typing import List
from typing import Optional

from geojson_pydantic import Feature
from geojson_pydantic.geometries import Geometry
from geojson_pydantic.geometries import parse_geometry_obj
from pydantic import BaseModel
from pydantic import validator
from pydantic.color import Color
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads as wkt_loads

from pado.images.ids import ImageId

# === PADO annotation model ===


class AnnotationState(enum.IntEnum):
    NOT_SET = -1
    PLANNED = 0
    ASSIGNED = 1
    IN_PROGRESS = 2
    DONE = 3
    REVIEWED = 4


class AnnotationQuality(str, enum.Enum):
    NOT_SET = "not_set"
    BROKEN = "broken"
    GOOD = "good"


class AnnotatorType(str, enum.Enum):
    HUMAN = "human"
    MODEL = "model"
    UNKNOWN = "unknown"


class Annotator(BaseModel):
    type: AnnotatorType
    name: str


# add a default instance thats not part of the model in case the annotator
# is unknown. NOTE: could have done metaclass, or descriptor but assignment
# to cls is easiest...
Annotator.UNKNOWN = Annotator(type=AnnotatorType.UNKNOWN, name="unknown")


class AnnotationStyle(str, enum.Enum):
    NOT_SET = "not_set"
    EXPLICIT_ROI_ONLY = "explicit_roi_only"
    ALL_WITHIN_REGION = "all_within_region"


class AnnotationModel(BaseModel):
    image_id: Optional[ImageId]
    identifier: Optional[str] = None
    project: Optional[str] = None
    annotator: Annotator = Annotator.UNKNOWN
    state: AnnotationState = AnnotationState.NOT_SET
    style: AnnotationStyle = AnnotationStyle.NOT_SET

    classification: str
    color: Color

    description: Optional[str]
    comment: str

    geometry: BaseGeometry

    class Config:
        arbitrary_types_allowed = True  # needed for geometry to work

    @validator("image_id", pre=True)
    def image_id_from_str(cls, v):
        if isinstance(v, str):
            return ImageId.from_str(v)
        return v

    @validator("geometry", pre=True)
    def geometry_from_str(cls, v):
        if isinstance(v, str):
            return wkt_loads(v)
        return v


# === QUPATH geojson annotations ===


class QPPathObjectId(str, enum.Enum):
    """these have been used as `.id` in legacy qupath<0.3"""

    ANNOTATION = "PathAnnotationObject"
    CELL = "PathCellObject"
    DETECTION = "PathDetectionObject"
    ROOT = "PathRootObject"
    TILE = "PathTileObject"
    TMA_CORE = "TMACoreObject"


class QPPathObjectType(str, enum.Enum):
    """these are used in `.properties.object_type` qupath>=0.3"""

    ANNOTATION = "annotation"
    CELL = "cell"
    DETECTION = "detection"
    ROOT = "root"
    TILE = "tile"
    TMA_CORE = "tma_core"


class QPClassification(BaseModel):
    """qupath classifications with colors"""

    name: str
    colorRGB: Color

    @validator("colorRGB", pre=True)
    def qupath_color_to_rgba(cls, v):
        """convert a qupath color to a pydantic Color"""
        if not (isinstance(v, int) and -(2**31) <= v <= 2**31 - 1):
            raise ValueError(f"color out of bounds: {v}")
        alpha, red, blue, green = struct.pack(">i", v)
        return Color((red, blue, green, alpha / 255.0))


class QPProperties(BaseModel):
    """qupath properties"""

    classification: QPClassification
    isLocked: bool
    measurements: Optional[List]  # todo: add measurement type?
    object_type: QPPathObjectType = None  # if provided must be of type
    name: str = None
    color: Color = None


class QuPathAnnotation(Feature[Geometry, QPProperties]):
    """model for qupath annotations"""

    id: QPPathObjectId = None

    @validator("geometry", pre=True)
    def parse_geometry_type(cls, v):
        if isinstance(v, dict):
            if "type" in v and v["type"] == "Polygon":
                # workaround for Polygons that do not adhere to geojson RFC 7946
                # fixme: I'm not sure if this should not be handled here...
                v = shape(v).__geo_interface__
            return parse_geometry_obj(v)
        return v
