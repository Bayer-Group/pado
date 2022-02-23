from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np
import pyvips

from pado.io.files import urlpathlike_to_fsspec
from pado.types import UrlpathLike


def create_image_prediction_tiff(
    input_data: np.ndarray | UrlpathLike,
    output_urlpath: UrlpathLike,
    *,
    tile_size: int = 256,
    input_storage_options: dict[str, Any] | None = None,
    output_storage_options: dict[str, Any] | None = None,
) -> None:
    """helper for writing an image_prediction tiff file"""
    with ExitStack() as stack:
        if isinstance(input_data, np.ndarray):
            # map np dtypes to vips
            dtype_to_format = {
                "uint8": "uchar",
                "int8": "char",
                "uint16": "ushort",
                "int16": "short",
                "uint32": "uint",
                "int32": "int",
                "float32": "float",
                "float64": "double",
                "complex64": "complex",
                "complex128": "dpcomplex",
            }

            def numpy2vips(a):
                height, width, bands = a.shape
                linear = a.reshape(width * height * bands)
                vi = pyvips.Image.new_from_memory(
                    linear.data, width, height, bands, dtype_to_format[str(a.dtype)]
                )
                return vi

            image = numpy2vips(input_data)
        else:
            buffer = stack.enter_context(
                urlpathlike_to_fsspec(input_data, storage_options=input_storage_options)
            )
            image = pyvips.Image.new_from_buffer(buffer, "", fail=True)

        with urlpathlike_to_fsspec(
            output_urlpath,
            mode="wb",
            storage_options=output_storage_options,
        ) as f:
            data = image.write_to_buffer(
                ".tiff",
                pyramid=True,
                tile=True,
                compression="jpeg",
                tile_width=tile_size,
                tile_height=tile_size,
            )
            f.write(data)
