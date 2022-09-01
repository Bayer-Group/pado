from __future__ import annotations

import json
import textwrap
import warnings
from collections import defaultdict
from typing import Callable
from typing import Optional

from shapely.geometry import shape

from pado.annotations.annotation import Annotation
from pado.annotations.annotation import Annotations
from pado.annotations.formats import AnnotationModel
from pado.annotations.formats import AnnotationState
from pado.annotations.formats import AnnotationStyle
from pado.annotations.formats import Annotator
from pado.annotations.formats import QuPathAnnotation
from pado.io.files import uncompressed
from pado.io.files import urlpathlike_to_fsspec
from pado.types import UrlpathLike

AnnotationsFromFileFunc = Callable[[UrlpathLike], Optional[Annotations]]


def qupath_geojson_annotations_loader(
    urlpath: UrlpathLike,
    *,
    identifier: Optional[str] = None,
    project: Optional[str] = None,
    annotator: Annotator = Annotator.UNKNOWN,
    state: AnnotationState = AnnotationState.NOT_SET,
    style: AnnotationStyle = AnnotationStyle.NOT_SET,
) -> Optional[Annotations]:
    """load a qupath geojson file as Annotations"""
    open_file = urlpathlike_to_fsspec(urlpath)

    with uncompressed(open_file) as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"not a geojson file: {open_file.path!r}")

    if isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            data = data["features"]
        elif "annotations" in data:
            data = data["annotations"]
    if not isinstance(data, list):
        raise ValueError(f"expected list: data={data!r}")

    records = []
    errors = defaultdict(list)
    missing_classifications = 0
    for feature in data:
        try:
            _ = feature.pop("id")
            a_qp = QuPathAnnotation(**feature)
        except BaseException as e:
            geom = feature.get("geometry", {})
            if (
                geom.get("type") == "Polygon"
                and len(geom.get("coordinates", [[]])[0]) < 3
            ):
                # FIXME: this should be handled somewhere else...
                pass  # skip
            elif "classification" not in feature["properties"]:
                missing_classifications += 1
            else:
                errors[(type(e).__name__, str(e))].append(feature)
            continue
        a_pd = AnnotationModel(
            image_id=None,
            identifier=identifier,
            project=project,
            annotator=annotator,
            state=state,
            style=style,
            classification=a_qp.properties.classification.name,
            color=a_qp.properties.classification.colorRGB,
            description=a_qp.properties.name or "",
            comment="",  # todo: this could be QuPath's description field, but it's not exported
            geometry=shape(a_qp.geometry),
        )
        records.append(Annotation(a_pd).to_record())

    if missing_classifications > 0:
        warnings.warn(
            f"File at {urlpath} contained {missing_classifications} missing classifications.",
            RuntimeWarning,
        )
    if errors:
        msg = ["PYDANTIC PARSING ERROR"]
        for (err_type, err_msg), features in errors.items():
            msg.extend(
                [
                    f"# {len(features)} OF ERROR {err_type}:",
                    f"{textwrap.indent(err_msg, prefix='# >>> ')}",
                    "# EXAMPLE:",
                    f"{textwrap.indent(json.dumps(features[0], indent=2), prefix='  ')}",
                ]
            )
        msg.append(f"# encountered {sum(map(len, errors.values()))} parsing errors")
        raise ValueError("\n".join(msg))

    return Annotations.from_records(records)
