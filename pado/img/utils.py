from typing import Tuple


def scale_xy(xy, *, current, target):
    x, y = xy
    mx0, my0 = current
    mx1, my1 = target
    return x * mx0 / mx1, y * my0 / my1


def tuple_round(xy: Tuple[float, float]) -> Tuple[int, int]:
    x, y = xy
    return int(round(x)), int(round(y))


def mpp(x, y=None):
    return float(x), float(y or x)
