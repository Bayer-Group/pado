import itertools

import pytest

from pado.structure import PadoColumn, PadoInvalid, PadoReserved, verify_columns


def test_column_enums():
    # test iterable and str
    for v in itertools.chain(PadoColumn, PadoReserved, PadoInvalid):
        assert v == str(v)


@pytest.mark.parametrize(
    "subcolumn",
    [
        "SUB__COLUMN",
        "NOT-CORRECT",
        "_WRONG_",
        str(PadoReserved.DATA_SOURCE_ID),
        str(PadoInvalid.RESERVED_COL_INDEX),
    ],
    ids=["dunder", "charset", "leading_", "reserved", "invalid"],
)
def test_subcolumn_allowed_values(subcolumn):
    # noinspection PyProtectedMember
    from pado.structure import SEPARATOR

    assert SEPARATOR == "__"

    with pytest.raises(ValueError):
        PadoColumn.IMAGE.subcolumn(subcolumn)


def test_verify_columns():

    # reserved columns are accepted
    assert verify_columns([PadoReserved.DATA_SOURCE_ID]) is True

    with pytest.raises(ValueError):
        verify_columns([PadoInvalid.RESERVED_COL_INDEX])

    with pytest.raises(ValueError):
        verify_columns(["NOT_A_TOPLEVEL"])

    with pytest.raises(ValueError):
        verify_columns(["IMAGE___WRONG"])

    assert verify_columns(["NOT_A_TOPLEVEL"], raise_if_invalid=False) is False
