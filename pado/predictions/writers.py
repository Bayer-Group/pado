from __future__ import annotations

import json
import os
import uuid
from contextlib import ExitStack
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pyvips
import zarr.hierarchy
import zarr.storage

from pado.images import Image
from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_fsspec
from pado.predictions.providers import ImagePrediction
from pado.predictions.providers import ImagePredictionProvider
from pado.predictions.providers import ImagePredictionType

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from pado.dataset import PadoDataset
    from pado.images import ImageId
    from pado.images.utils import Bounds
    from pado.images.utils import IntSize
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


class ImagePredictionWriter:
    """image prediction writer

    let's you incrementally build image predictions

    >>> writer = ImagePredictionWriter(...)
    >>> for tile in range(dataset):
    >>>     output = predict(tile)
    >>>     bounds = get_bounds(tile)
    >>>     writer.add_prediction(output, bounds=bounds)
    >>> writer.store_in_dataset(...)

    """

    def __init__(
        self,
        *args,
        extra_metadata: dict[str, Any],
        image_id: ImageId | None = None,
        **kwargs,
    ) -> None:
        self._image_id: ImageId | None = image_id
        self._size_map: dict[ImageId, IntSize] = {}

        store = zarr.storage.TempStore(prefix="pado_zarr", normalize_keys=True)
        group = zarr.hierarchy.group(store=store)
        self._group = group
        self._chunk_size: tuple[int] | None = None
        self._output_dtype: np.dtype | None = None
        self._fill_value: float = 0

        assert isinstance(extra_metadata, dict) and json.dumps(extra_metadata)
        self._extra_metadata = extra_metadata

    def set_size_map(self, smap: dict[ImageId, IntSize]):
        self._size_map = smap

    def set_input(
        self,
        *,
        image_id: ImageId,
        image_size: IntSize | None = None,
    ):
        self._image_id = image_id
        if image_size is None:
            assert image_id in self._size_map
        else:
            self._size_map[image_id] = image_size

    def set_output(
        self,
        *,
        tile_shape: tuple[int],
        tile_dtype: np.dtype,
        fill_value: float = 0,
    ):
        s = tuple(tile_shape)
        assert len(s) == 3 and all(isinstance(x, int) and x > 0 for x in s)
        self._chunk_size = s
        self._output_dtype = tile_dtype
        self._fill_value = fill_value

    def get_zarr_array(self, image_id: ImageId) -> zarr.Array:
        name = image_id.to_url_id()
        try:
            return self._group[name]
        except KeyError:
            width, height = self._size_map[image_id].as_tuple()
            cheight, cwidth, cdepth = self._chunk_size
            return self._group.create(
                name=name,
                shape=(height, width, cdepth),
                chunks=self._chunk_size,
                dtype=self._output_dtype,
                fill_value=self._fill_value,
            )

    def add_prediction(self, prediction_data: ArrayLike, *, bounds: Bounds) -> None:
        """add a tile prediction to the writer"""
        assert self._image_id is not None  # todo: lift restriction
        assert prediction_data.shape == self._chunk_size
        assert (
            self._size_map[self._image_id].mpp == bounds.mpp
        )  # todo: lift restriction
        arr = self.get_zarr_array(self._image_id)
        arr[
            int(bounds.y_left) : int(bounds.y_right),
            int(bounds.x_left) : int(bounds.x_right),
            :,
        ] = prediction_data

    def store_in_dataset(
        self, ds: PadoDataset, *, predictions_path: str = "../predictions"
    ) -> None:
        """store the collected prediction data in the pado dataset"""
        assert not ds.readonly

        ipp = {}

        def _get_image_prediction_urlpath(n):
            pth = ds._ensure_dir(predictions_path)
            return os.path.join(pth, f"{n}-{uuid.uuid4()}.tif")

        for image_id in self._size_map:
            name = image_id.to_url_id()
            if name not in self._group:
                continue

            arr = self.get_zarr_array(image_id)
            urlpath = fsopen(ds._fs, _get_image_prediction_urlpath(name))
            create_image_prediction_tiff(
                arr[:],
                urlpath,
            )
            pred = Image(urlpath)

            ipp[image_id] = [
                ImagePrediction(
                    image_id=image_id,
                    prediction_type=ImagePredictionType.HEATMAP,
                    bounds=Bounds(
                        0,
                        0,
                        pred.dimensions.x,
                        pred.dimensions.y,
                        mpp=self._size_map[image_id].mpp,
                    ),
                    extra_metadata=self._extra_metadata,
                    image=pred,
                )
            ]

        provider = ImagePredictionProvider(ipp, identifier="aignostics")
        ds.ingest_obj(provider)
