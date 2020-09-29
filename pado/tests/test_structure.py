import itertools

import pytest

from pado.structure import (
    PadoColumn,
    PadoInvalid,
    PadoReserved,
    build_column_map,
    structurize_metadata,
    verify_columns,
)


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


def test_build_column_map(datasource):
    with datasource:
        m = build_column_map(datasource.metadata.columns)


def test_build_column_map_raise_missing():
    with pytest.raises(ValueError):
        build_column_map(["UNKNOWN_TOPLEVEL"])


def test_structurize_metadata(datasource):
    with datasource:
        col_map = build_column_map(datasource.metadata.columns)
        # prepare dataframe
        df = datasource.metadata
    img_id = df[PadoColumn.IMAGE].iloc[0]
    sdf = df.loc[df[PadoColumn.IMAGE] == img_id, :]
    # structure
    s = structurize_metadata(sdf, PadoColumn.IMAGE, col_map)
    assert s
