from __future__ import annotations

import json
import os
import random
import uuid
import warnings
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import numpy as np
import pyvips
import zarr.hierarchy
import zarr.storage

from pado.images import Image
from pado.images import ImageId
from pado.images.utils import MPP
from pado.images.utils import Bounds
from pado.images.utils import IntBounds
from pado.images.utils import IntSize
from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_fsspec
from pado.predictions.providers import ImagePrediction
from pado.predictions.providers import ImagePredictionProvider
from pado.predictions.providers import ImagePredictionType

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from numpy.typing import NDArray

    from pado.dataset import PadoDataset
    from pado.types import UrlpathLike

__all__ = [
    "ImagePredictionWriter",
]


def create_image_prediction_tiff(
    input_data: np.ndarray | UrlpathLike,
    output_urlpath: UrlpathLike,
    *,
    tile_size: int = 256,
    input_storage_options: dict[str, Any] | None = None,
    output_storage_options: dict[str, Any] | None = None,
    mpp: MPP | None = None,
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
            image = pyvips.Image.new_from_buffer(
                buffer, "", access="sequential", fail=True
            )

        kw = {}
        if mpp:
            kw["xres"] = 1000.0 / mpp.x  # pixels per millimeter
            kw["yres"] = 1000.0 / mpp.y

        with urlpathlike_to_fsspec(
            output_urlpath,
            mode="wb",
            storage_options=output_storage_options,
        ) as f:
            data = image.write_to_buffer(
                ".tiff",
                pyramid=True,
                tile=True,
                subifd=False,
                bigtiff=True,
                # properties=True,
                compression="jpeg",
                tile_width=tile_size,
                tile_height=tile_size,
                **kw,
            )
            f.write(data)


def _multichannel_to_rgb(
    arr: ArrayLike,
    *,
    channel_colors: Sequence[tuple[int, int, int]],
    single_channel_threshold: float | None = None,
) -> NDArray[np.uint8]:
    """converts a (single/multi)-channel array (X, Y, N) to an RGB color array of size (X, Y, 3)

    Notes
    -----
      We enforce that one channel is (255, 255, 255) white color coded.
      In the single channel case that is used as the background color.

    Parameters
    ----------
    channel_colors:
        one rgb color per channel
    single_channel_threshold:
        if single channel array provided to distinguish background from foreground

    """
    BACKGROUND_COLOR = (255, 255, 255)
    # assume:
    # - every channel stores some sort of score/probability for a single class
    # -
    assert arr.ndim == 3
    num_channels = arr.shape[2]

    colors = np.array(channel_colors, dtype=np.uint8)

    assert colors.ndim == 2 and colors.shape[1] == 3, "requires rgb colors"
    if num_channels != colors.shape[0]:
        raise ValueError("need as many colors as channels")

    if num_channels == 1:
        assert (
            single_channel_threshold is not None
        ), "single channel requires to define a single_channel_threshold"
        arr = (arr > single_channel_threshold).astype(int)[:, :, 0]
        colors = np.array([BACKGROUND_COLOR, colors[0]], dtype=np.uint8)

    else:
        assert (
            single_channel_threshold is None
        ), "multi channel will just use the max across channels"
        assert BACKGROUND_COLOR in colors
        arr = np.argmax(arr, axis=2)

    # todo:
    # - implement morphological filtering
    # - smoothing, etc...
    return colors[arr]


class ImagePredictionWriter:
    """image prediction writer

    let's you incrementally build image predictions

    >>> writer = ImagePredictionWriter(extra_metadata={"model": "my-model", "something": 4})
    >>> writer.set_input(image_id=my_id, image_size=size_of_original)  # IntSize(W, H, mpp=mpp0)
    >>> writer.set_output(
    ...     tile_shape=(512, 512, 4), tile_dtype=np.float64, fill_value=0.0,
    ...     channel_colors=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 255)],
    ... )
    >>> for tile in range(dataset):
    >>>     output = predict(tile)
    >>>     bounds = get_bounds(tile)
    >>>     writer.add_prediction(output, bounds=bounds)
    >>> writer.store_in_dataset(...)

    """

    def __init__(
        self,
        *,
        extra_metadata: dict[str, Any],
        identifier: str | None = None,
        raise_on_overwrite: bool = True,
        **_kwargs,
    ) -> None:
        """instantiate the ImagePredictionWriter

        Parameters
        ----------
        extra_metadata:
            a json serializable metadata dict. Must be provided. Currently used
            in our visualization tools to carry information about the model and
            the specific runs.
        identifier:
            used to name the image prediction provider. If None, a UUID is
            created to minimize name clashes.
        raise_on_overwrite:
            raise an error if add_prediction would overwrite already provided
            data. Useful to ensure that the predictions don't overlap.

        """
        self._image_id: ImageId | None = None
        self._size_map: dict[ImageId, IntSize] = {}
        self._raise_on_overwrite = raise_on_overwrite
        self._mask_map: dict[ImageId, set[tuple[int, int, int, int]]] = defaultdict(set)

        if identifier is None:
            self._identifier = uuid.uuid4()
        else:
            self._identifier = identifier

        store = zarr.storage.TempStore(prefix="pado_zarr", normalize_keys=True)
        group = zarr.hierarchy.group(store=store)
        self._group = group
        self._chunk_size: tuple[int] | None = None
        self._output_dtype: np.dtype | None = None
        self._fill_value: float = 0
        self._is_rgb_prediction: bool | None = None

        # color conversion kwargs
        self._color_conversion_kwargs = {}

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
        tile_shape: tuple[int, ...],
        tile_dtype: np.dtype | type[np.numeric],
        fill_value: float = 0,
        is_rgb_prediction: bool = False,
        channel_colors: Sequence[tuple[int, int, int]] = (),
        single_channel_threshold: float | None = None,
    ):
        """set_output defines the image output

        Parameters
        ----------
        tile_shape:
            the shape of the tile as a tuple. Currently, requires len(tile_shape) == 3.
        tile_dtype:
            the numpy dtype of the prediction tile.
        fill_value:
            used to fill in empty chunks of the underlying zarr Array
        is_rgb_prediction:
            set to True if you provide an RGB tile in `.add_prediction`
        channel_colors:
            if you provide prediction scores in a multichannel tile, each channel requires
            a color. provide as tuple[int, int, int] with ranges 0 <= c <= 255
        single_channel_threshold:
            if you provide prediction scores in a singlechannel tile, set this threshold
            to distinguish between positive and negative predictions.

        """
        s = tuple(tile_shape)
        assert len(s) == 3 and all(isinstance(x, int) and x > 0 for x in s)
        self._chunk_size = s
        self._output_dtype = tile_dtype

        self._is_rgb_prediction = bool(is_rgb_prediction)
        if not is_rgb_prediction:
            self._fill_value = fill_value

        else:
            assert s[2] == 3, f"not an RBG tile: {s!r} need 3 color channels"
            assert tile_dtype == np.uint8, f"needs to be uint8 dtype, got: {tile_dtype}"
            assert channel_colors == ()
            assert single_channel_threshold is None

        self._color_conversion_kwargs["channel_colors"] = channel_colors
        self._color_conversion_kwargs[
            "single_channel_threshold"
        ] = single_channel_threshold

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

    def add_prediction(self, prediction_data: ArrayLike, *, bounds: IntBounds) -> None:
        """add a tile prediction to the writer"""
        assert self._image_id is not None  # todo: lift restriction

        iid = self._image_id
        size = self._size_map[iid]
        arr = self.get_zarr_array(iid)

        if not isinstance(bounds, IntBounds):
            warnings.warn(
                "Received a Bounds instance instead of IntBounds. "
                "Doublecheck why you're currently using float coordinates!"
            )
            bounds = bounds.floor()

        assert size.mpp == bounds.mpp  # todo: lift restriction

        x0, y0, x1, y1 = b = tuple(map(int, bounds.as_tuple()))

        if self._raise_on_overwrite:
            # test that we don't overwrite existing data
            written_bounds = self._mask_map[iid]
            if any(
                (x1 > _x0 and _x1 > x0) and (y1 > _y0 and _y1 > y0)
                for _x0, _y0, _x1, _y1 in written_bounds
            ):
                raise RuntimeError(
                    f"bounds: {b!r} overlap with already set bounds:\n{written_bounds!r}"
                )
            else:
                # could merge adjacent tiles if this becomes a bottleneck
                written_bounds.add(b)

        assert (
            prediction_data.shape == self._chunk_size
        ), f"{prediction_data.shape!r} == {self._chunk_size!r}"

        ah, aw = arr.shape[:2]
        # todo: manage in Bounds class
        x1 = min(x1, aw)
        y1 = min(y1, ah)
        arr[y0:y1, x0:x1, :] = prediction_data[: (y1 - y0), : (x1 - x0), :]

    def store_in_local_dir(
        self,
        path: os.PathLike,
        *,
        predictions_path: str = "_image_predictions",
    ):
        """store the predictions in a local dir"""
        ipp = {}

        pth = Path(path)
        if not pth.is_dir():
            pth.mkdir(parents=True)

        def _get_image_prediction_urlpath(n):
            p = pth.joinpath(predictions_path)
            p.mkdir(exist_ok=True)
            return p.joinpath(f"{n}-{uuid.uuid4()}.tif")

        for image_id in self._size_map:
            name = image_id.to_url_id()
            if name not in self._group:
                continue

            arr = self.get_zarr_array(image_id)
            urlpath = _get_image_prediction_urlpath(name)
            if self._is_rgb_prediction:
                assert arr.ndim == 3 and arr.shape[2] == 3
                rgb_arr = arr[:]
            else:
                rgb_arr = _multichannel_to_rgb(arr[:], **self._color_conversion_kwargs)

            create_image_prediction_tiff(
                rgb_arr,
                urlpath,
                mpp=self._size_map[image_id].mpp,
            )
            pred = Image(urlpath, load_metadata=True, load_file_info=True)

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

        provider = ImagePredictionProvider(ipp, identifier=self._identifier)
        provider.to_parquet(
            pth.joinpath(f"{self._identifier}.image_predictions.parquet")
        )

    def store_in_dataset(
        self, ds: PadoDataset, *, predictions_path: str = "../predictions"
    ) -> None:
        """store the collected prediction data in the pado dataset"""
        assert not ds.readonly

        ipp = {}

        def _get_image_prediction_urlpath(n):
            pth = ds._ensure_dir(predictions_path)
            return os.path.join(pth, f"{n}-{uuid.uuid4()}.tif")  # fixme: normalize path

        for image_id in self._size_map:
            name = image_id.to_url_id()
            if name not in self._group:
                continue

            arr = self.get_zarr_array(image_id)
            urlpath = fsopen(ds._fs, _get_image_prediction_urlpath(name), mode="wb")
            if self._is_rgb_prediction:
                assert arr.ndim == 3 and arr.shape[2] == 3
                rgb_arr = arr[:]
            else:
                rgb_arr = _multichannel_to_rgb(arr[:], **self._color_conversion_kwargs)

            create_image_prediction_tiff(
                rgb_arr,
                urlpath,
                mpp=self._size_map[image_id].mpp,
            )
            pred = Image(urlpath, load_metadata=True, load_file_info=True)

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

        provider = ImagePredictionProvider(ipp, identifier=self._identifier)
        ds.ingest_obj(provider)


if __name__ == "__main__":

    print("writing test image")
    H, W, C = 10000, 8000, 5
    TW = TH = 512
    colors = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
    ]

    writer = ImagePredictionWriter(extra_metadata={})
    iid = ImageId("test")
    size = IntSize(W, H, mpp=MPP(1.0, 1.0))
    writer.set_input(image_id=iid, image_size=size)
    writer.set_output(
        tile_shape=(TH, TW, C),
        tile_dtype=np.float32,
        fill_value=0.0,
        channel_colors=colors,
    )

    for x in range(0, W, TW):
        for y in range(0, H, TH):
            b = Bounds.from_tuple((x, y, min(x + TW, W), min(y + TH, H)), mpp=size.mpp)
            arr = np.zeros((TH, TW, C)).astype(np.float32)
            arr[:, :, random.randint(0, C - 1)] = 1.0
            writer.add_prediction(arr, bounds=b)

    writer.store_in_local_dir(".", predictions_path=".")
