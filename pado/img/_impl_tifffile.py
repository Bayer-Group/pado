import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple

import numpy as np
from tifffile import TIFF
from tifffile import TiffPage
from tifffile.tifffile import svs_description_metadata

from pado.img._base import ImageBackend
from pado.img._base import UnsupportedImageFormat
from pado.img._base import N
from pado.img.utils import mpp
from pado.img.utils import scale_xy
from pado.img.utils import tuple_round

try:
    from tifffile import TiffFile
    from tifffile import TiffFileError
except ImportError:
    TiffFile = TiffFileError = None


class TiffFileImageBackend(ImageBackend):

    def __init__(self, path):
        super().__init__(path)
        self._slide: Optional[TiffFile] = None
        self._stack = None

        self._slide_levels: Optional[Dict[int, TiffPage]] = None
        self._level_info = None
        self._metadata = None

    @property
    def path(self) -> Path:
        return self._path

    def open(self):
        if self._slide is not None:
            return
        self._stack = ExitStack()
        try:
            self._slide = self._stack.enter_context(TiffFile(os.fspath(self._path)))
        except TiffFileError as err:
            raise UnsupportedImageFormat(f"{self._path}: {err}")

        try:
            # get Baseline series
            for series in self._slide.series:
                if series.name == "Baseline":
                    break
            else:
                raise UnsupportedImageFormat(f"{self._path}: no baseline series")
            assert series.is_pyramidal

            self._slide_levels = {}
            for lvl_idx, lvl in enumerate(series.levels):
                page, = self._slide_levels[lvl_idx], = lvl.pages  # unpack single page into dict
                # noinspection PyUnresolvedReferences
                assert (
                        page.compression == TIFF.COMPRESSION.JPEG
                        and page.is_tiled
                        and page.planarconfig == TIFF.PLANARCONFIG.CONTIG
                )

        except Exception:
            # cleanup if we error while initializing
            self.close()
            raise

    def close(self):
        if self._stack is None:
            return
        self._slide_levels = None
        self._stack.close()
        self._stack = self._slide = None

    def _get_level_info(self):
        if self._level_info is None:

            level_dimensions = []
            tile_sizes = []
            lvl0 = self._slide_levels[0]
            for _, page in sorted(self._slide_levels.items()):
                level_dimensions.append((page.imagewidth, page.imagelength))
                tile_sizes.append((page.tilewidth, page.tilelength))

            downsamples = []
            i = 0 if level_dimensions[0][0] > level_dimensions[0][1] else 1
            for dim in level_dimensions:
                downsamples.append(level_dimensions[0][i] / dim[i])

            self._level_info = (
                lvl0.description, level_dimensions, tile_sizes, downsamples
            )

        return self._level_info

    def image_metadata(self) -> Mapping[str, Any]:
        """load image metadata identical to openslide"""
        if self._metadata is None:
            aperio_description, lvl_dimensions, tile_sizes, downsamples = self._get_level_info()

            aperio_meta = svs_description_metadata(aperio_description)
            aperio_meta.pop("")
            aperio_meta.pop("Aperio Image Library")

            ad = {
                f"aperio.{k}": v for k, v in sorted(aperio_meta.items())
            }
            od = {
                "openslide.level-count": len(downsamples)
            }
            for lvl, (ds, (width, height), (tile_width, tile_height)) in enumerate(zip(
                downsamples, lvl_dimensions, tile_sizes
            )):
                od[f"openslide.level[{lvl}].downsample"] = ds
                od[f"openslide.level[{lvl}].height"] = height
                od[f"openslide.level[{lvl}].width"] = width
                od[f"openslide.level[{lvl}].tile-height"] = tile_height
                od[f"openslide.level[{lvl}].tile-width"] = tile_width
            td = {
                "tiff.ImageDescription": aperio_description,
            }

            md = {
                'width': lvl_dimensions[0][0],
                'height': lvl_dimensions[0][1],
                'objective_power': aperio_meta["AppMag"],
                'mpp_x': aperio_meta["MPP"],
                'mpp_y': aperio_meta["MPP"],
                'downsamples': downsamples,
                'vendor': "aperio",
                # Optional props
                'background_color': None,
                'quickhash1': None,
                'slide_comment': aperio_description,
                'bounds_x': None,
                'bounds_y': None,
                'bounds_width': None,
                'bounds_height': None,
            }

            # add all other unused keys to the dictionary prefixed with "_"
            md.update(**ad, **od, **td)
            self._metadata = md

        return self._metadata

    @property
    def level0_mpp(self) -> Tuple[float, float]:
        return (
            self.image_metadata()["aperio.MPP"],
            self.image_metadata()["aperio.MPP"],
        )

    @property
    def level_mpp_map(self):
        mpp_x, mpp_y = self.level0_mpp
        return {
            lvl: mpp(mpp_x * ds, mpp_y * ds)
            for lvl, ds in enumerate(self.image_metadata()["downsamples"])
        }

    def get_size(self, level: int = 0) -> Tuple[int, int]:
        return self._get_level_info()[1][0]

    def get_region(
            self,
            location_xy: Tuple[int, int],
            region_wh: Tuple[int, int],
            level: int = 0,
    ) -> np.array:
        page = self._slide_levels[level]



