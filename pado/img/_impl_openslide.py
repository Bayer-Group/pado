import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple

import numpy as np
import openslide
from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError

from pado.img._base import ImageBackend
from pado.img._base import UnsupportedImageFormat
from pado.img._base import N
from pado.img.utils import mpp
from pado.img.utils import scale_xy
from pado.img.utils import tuple_round


class OpenSlideImageBackend(ImageBackend):

    _openslide_cls = OpenSlide

    def __init__(self, path):
        if self._openslide_cls is None:
            raise RuntimeError("openslide could not be imported")
        super().__init__(path)
        self._slide: Optional[OpenSlide] = None
        self._stack = None

    @property
    def path(self) -> Path:
        return self._path

    def open(self):
        if self._slide is not None:
            return
        self._stack = ExitStack()
        try:
            self._slide = self._stack.enter_context(self._openslide_cls(os.fspath(self._path)))
        except OpenSlideUnsupportedFormatError as err:
            raise UnsupportedImageFormat(f"{self._path}: {err}")

    def close(self):
        if self._stack is None:
            return
        self._stack.close()
        self._stack = self._slide = None

    def image_metadata(self) -> Mapping[str, Any]:
        dimensions = self._slide.dimensions
        props = self._slide.properties
        # keep track of used keys
        _used_keys = {}
        def _props_get(os_key): return _used_keys.setdefault(os_key, props.get(os_key))

        md = {
            'width': dimensions[0],
            'height': dimensions[1],
            'objective_power': _props_get(openslide.PROPERTY_NAME_OBJECTIVE_POWER),
            'mpp_x': _props_get(openslide.PROPERTY_NAME_MPP_X),
            'mpp_y': _props_get(openslide.PROPERTY_NAME_MPP_Y),
            'downsamples': list(self._slide.level_downsamples),
            'vendor': _props_get(openslide.PROPERTY_NAME_VENDOR),
            # Optional props
            'background_color': _props_get(openslide.PROPERTY_NAME_BACKGROUND_COLOR),
            'quickhash1': _props_get(openslide.PROPERTY_NAME_QUICKHASH1),
            'slide_comment': _props_get(openslide.PROPERTY_NAME_COMMENT),
            'bounds_x': _props_get(openslide.PROPERTY_NAME_BOUNDS_X),
            'bounds_y': _props_get(openslide.PROPERTY_NAME_BOUNDS_Y),
            'bounds_width': _props_get(openslide.PROPERTY_NAME_BOUNDS_WIDTH),
            'bounds_height': _props_get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT),
        }
        assert set(vars(N).values()).issubset(md), f"missing mandatory keys '{set(vars(N).values()) - set(md)}'"

        # add all other unused keys to the dictionary prefixed with "_"
        for key, value in sorted(props.items()):
            if key in _used_keys:
                continue
            md[key] = value

        return md

    @property
    def level0_mpp(self) -> Tuple[float, float]:
        return mpp(
            self._slide.properties[openslide.PROPERTY_NAME_MPP_X],
            self._slide.properties[openslide.PROPERTY_NAME_MPP_Y],
        )

    @property
    def level_mpp_map(self):
        mpp_x, mpp_y = mpp(
            self._slide.properties[openslide.PROPERTY_NAME_MPP_X],
            self._slide.properties[openslide.PROPERTY_NAME_MPP_Y],
        )
        return {
            lvl: mpp(mpp_x * ds, mpp_y * ds)
            for lvl, ds in enumerate(self._slide.level_downsamples)
        }

    def get_size(self, level: int = 0) -> Tuple[int, int]:
        return self._slide.level_dimensions[level]

    def get_region(
        self,
        location_xy: Tuple[int, int],
        region_wh: Tuple[int, int],
        level: int = 0,
        *,
        downsize_to: Optional[Tuple[int, int]] = None
    ) -> np.array:
        img = self._slide.read_region(location_xy, level, region_wh)
        # this is always a RGBA image...
        # let's throw away the alpha layer immediately in our abstraction
        if downsize_to:
            img.thumbnail(downsize_to)
        return np.array(img)[:, :, :3]
