from __future__ import annotations

import os
import pickle
from sys import float_info

import fsspec
import pytest

from pado.images.ids import ImageId
from pado.images.ids import load_image_ids_from_csv
from pado.images.providers import update_image_provider_urlpaths
from pado.images.utils import MPP
from pado.images.utils import IntPoint
from pado.images.utils import IntSize
from pado.images.utils import match_mpp
from pado.io.files import urlpathlike_to_fsspec

# --- test constructors -----------------------------------------------

IMAGE_ID_ARG_KWARG_LIST = [
    pytest.param([("a",), {}], id="single_part"),
    pytest.param([("a", "b", "c"), {}], id="multi_part"),
    pytest.param([("a b",), {}], id="part_with_space"),
    pytest.param([("a\u1234b",), {}], id="part_with_unicode"),
    pytest.param([("a", "b"), {"site": "mars"}], id="with_site"),
]


@pytest.mark.parametrize("image_id_args_kwargs", IMAGE_ID_ARG_KWARG_LIST)
def test_image_id___new__(image_id_args_kwargs):
    args, kwargs = image_id_args_kwargs
    _ = ImageId(*args, **kwargs)  # just test the constructor


@pytest.mark.parametrize(
    "image_id_args_kwargs",
    [
        pytest.param([(), {}], id="empty"),
        pytest.param([(1,), {}], id="non_string_only"),
        pytest.param([("a", 1), {}], id="non_string_some"),
        pytest.param([(["a", "b"],), {}], id="iterable_at_args0"),
        pytest.param([("ImageId('a')",), {}], id="str_serialized_image_id"),
        pytest.param([('{"image_id":["a"]}',), {}], id="json_serialized_image_id"),
    ],
)
def test_image_id_incorrect_input(image_id_args_kwargs):
    args, kwargs = image_id_args_kwargs
    with pytest.raises(ValueError):
        ImageId(*args, **kwargs)


@pytest.mark.parametrize(
    "image_id_arg_site",
    [
        pytest.param(["a", None], id="str"),
    ],
)
def test_image_id_make_incorrect_input(image_id_arg_site):
    arg, site = image_id_arg_site
    with pytest.raises(TypeError):
        ImageId.make(arg, site=site)


def test_image_id_make():
    iid = ImageId.make(["a", "b"], site="mars")
    assert iid == ImageId("a", "b", site="mars")


@pytest.fixture(scope="function")
def image_id(request):
    args, kwargs = request.param
    yield ImageId(*args, **kwargs)


@pytest.mark.parametrize("image_id", IMAGE_ID_ARG_KWARG_LIST, indirect=True)
def test_image_id_from_image_id(image_id):
    assert ImageId(image_id) == image_id


# --- test round tripping ---------------------------------------------


@pytest.mark.parametrize("image_id", IMAGE_ID_ARG_KWARG_LIST, indirect=True)
def test_image_id_pickle_roundtrip(image_id):
    serialized = pickle.dumps(image_id)
    new_id = pickle.loads(serialized)  # nosec B301
    assert image_id == new_id


@pytest.mark.parametrize("image_id", IMAGE_ID_ARG_KWARG_LIST, indirect=True)
def test_image_id_str_roundtrip(image_id):
    id_str = image_id.to_str()
    new_id = ImageId.from_str(id_str)
    assert image_id == new_id


def test_image_id_from_str_incorrect_input_type():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ImageId.from_str(1)


@pytest.mark.parametrize(
    "id_input",
    [
        pytest.param("", id="empty_str"),
        pytest.param("('a')", id="tuple_str"),
        pytest.param("ImageId('a'')", id="unparsable"),
        pytest.param('{"image_id":["a"]}', id="json"),
    ],
)
def test_image_id_from_str_incorrect_input(id_input):
    with pytest.raises(ValueError):
        _ = ImageId.from_str(id_input)


def test_image_id_from_str_with_subclass():
    # note: this is basically irrelevant... very unlikely anyone
    #   will subclass this ever. we could mark the class `final`
    #   but... *screams in questionable new python features*
    serialized = ImageId("a", "b", site="mars").to_str()

    class Why(ImageId):
        pass

    with pytest.raises(ValueError):
        Why.from_str(serialized)


@pytest.mark.parametrize("image_id", IMAGE_ID_ARG_KWARG_LIST, indirect=True)
def test_image_id_json_roundtrip(image_id):
    id_json = image_id.to_json()
    new_id = ImageId.from_json(id_json)
    assert image_id == new_id


@pytest.mark.parametrize("image_id", IMAGE_ID_ARG_KWARG_LIST, indirect=True)
def test_image_id_str_roundtrip_no_eval(image_id, monkeypatch):
    import pado.images.ids

    monkeypatch.setattr(pado.images.ids, "_PADO_BLOCK_IMAGE_ID_EVAL", True)
    id_str = image_id.to_str()
    new_id = ImageId.from_str(id_str)
    assert image_id == new_id


def test_image_id_from_json_incorrect_input_type():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ImageId.from_json(1)


@pytest.mark.parametrize(
    "id_input",
    [
        pytest.param("", id="empty_str"),
        pytest.param("ImageId('a')", id="str_id"),
        pytest.param('{"image_id":"filename.svs"}', id="incorrect_json"),
    ],
)
def test_image_id_from_json_incorrect_input(id_input):
    with pytest.raises(ValueError):
        _ = ImageId.from_json(id_input)


# --- test equality and hashing ---------------------------------------


def test_image_id_eq_dont_coerce():
    iid = ImageId("a", "b", site="mars")
    iid_tuple = ("mars", "a", "b")
    assert (iid == iid_tuple) is False
    assert iid != iid_tuple


def test_image_id_to_base64_encoding():
    iid = ImageId("a", "b", site="mars")
    iid_b64_encoded = "SW1hZ2VJZCgnYScsICdiJywgc2l0ZT0nbWFycycp"
    assert iid.to_url_id() == iid_b64_encoded


def test_image_id_current_hash_assumption():
    iid = ImageId("a", "b", site="mars")
    fn_tuple = ("b",)
    assert hash(iid) == hash(fn_tuple)
    assert iid != fn_tuple


def test_image_id_without_site_when_used_as_key():
    iid_with_site = ImageId("a", "b", site="mars")
    iid_no_site = ImageId("a", "b")
    wrong_iid_no_site = ImageId("a", "c")

    data = {iid_with_site: "metadata"}
    assert data.get(wrong_iid_no_site, None) is None
    assert data.get(iid_no_site, None) == "metadata"


# --- test path handling ----------------------------------------------


def test_image_id_site_when_mapper_not_available():
    iid = ImageId("a", "b", site="site-does-not-exist")

    with pytest.raises(KeyError):
        _ = iid.id_field_names
    with pytest.raises(KeyError):
        _ = os.fspath(iid)
    with pytest.raises(KeyError):
        _ = iid.to_path()


def test_image_id_file_mapper_id_field_names():
    iid = ImageId("a", "b")  # no site!
    assert iid.id_field_names == ("site", "filename")


def test_image_id_file_mapper_fspath():
    iid = ImageId("a", "b")  # no site!
    assert os.fspath(iid)


def test_image_id_file_mapper_to_path():
    iid = ImageId("a", "b")  # no site!
    assert iid.to_path()


def test_load_image_ids_from_csv(tmp_path):
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text("c0,c1\ni10,i11\n")
    tids, header = load_image_ids_from_csv(csv_file, csv_columns=["c1", "c0"])
    assert tids == [("i11", "i10")]
    assert header == ["c0", "c1"]


def test_load_image_ids_from_csv_no_header(tmp_path):
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text("i00,i01\ni10,i11\n")
    tids, header = load_image_ids_from_csv(csv_file, csv_columns=[1, 0], no_header=True)
    assert tids == [
        ("i01", "i00"),
        ("i11", "i10"),
    ]
    assert header is None


def test_load_image_ids_from_csv_wrong_column_type(tmp_path):
    csv_file = tmp_path.joinpath("iids.csv")
    with pytest.raises(TypeError) as e:
        load_image_ids_from_csv(csv_file, csv_columns=[1.0, 2.0])
        e.match("csv_columns must be a list[int] or list[str], got: list[float]")
    with pytest.raises(TypeError) as e:
        load_image_ids_from_csv(csv_file, csv_columns=1.0)
        e.match("csv_columns must be a list[int] or list[str], got: float")


# --- test in datasets ------------------------------------------------


def test_image_id_in_dataset(dataset_ro):
    key = next(iter(dataset_ro.images))
    assert isinstance(key, ImageId)
    assert key in dataset_ro.annotations
    assert key in dataset_ro.metadata


def test_update_image_provider_urlpaths(dataset, tmp_path, capsys):

    new_image_loc = tmp_path.joinpath("somewhere_else")
    new_image_loc.mkdir()
    for image_id in dataset.index[:-1]:
        image = dataset.images[image_id]
        with urlpathlike_to_fsspec(image.urlpath, mode="rb") as f:
            new_image_loc.joinpath(image_id.last).write_bytes(f.read())

    update_image_provider_urlpaths(
        new_image_loc,
        "*.svs",
        provider=dataset.images,
        progress=True,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "found 2 new files matching the pattern" in captured.err
    assert "provider has 3 images" in captured.err
    assert "trying to match new files" in captured.err
    assert "re-associated 2 images" in captured.err


def test_image_open(dataset):
    img = dataset[0].image
    img.open()
    try:
        assert img.get_array(IntPoint(0, 0), IntSize(10, 10), 0).sum() != 0
    finally:
        img.close()


def test_image_open_via_fs(dataset, tmp_path):
    img = dataset[0].image

    of = fsspec.open(f"simplecache::file:{tmp_path}")

    img.open(via=of.fs)
    try:
        assert img.get_array(IntPoint(0, 0), IntSize(10, 10), 0).sum() != 0
    finally:
        img.close()


def test_image_via(dataset):
    img = dataset[0].image
    img.via(dataset)
    try:
        assert img.get_array(IntPoint(0, 0), IntSize(10, 10), 0).sum() != 0
    finally:
        img.close()


def test_image_ctx_manager(dataset):
    with dataset[0].image as img:
        _ = img.metadata


def test_image_ctx_manager_via(dataset):
    with dataset[0].image.via(dataset) as img:
        _ = img.metadata


def test_image_get_chunk_sizes(dataset):
    for img in dataset.images.values():
        with img:
            chunk_sizes = img.get_chunk_sizes(level=0)
            assert chunk_sizes.ndim == 2


def test_mpp_equality():
    m0 = MPP(0.5, 0.5)
    m1 = MPP(0.51, 0.51)
    m2 = MPP(0.51, 0.51, rtol=0.05, atol=0)

    assert m0 != m1
    assert m1 != m0
    assert m0 == m2
    assert m2 == m0

    m0 = MPP(1, 1, rtol=1, atol=0)
    m1 = MPP(10, 10, rtol=0.1, atol=0)
    assert m0 != m1
    assert m1 != m0

    m0 = MPP(1, 1, rtol=1)
    m1 = MPP(2, 2)
    assert m0 == m1
    assert m1 == m0


def test_mpp_close_rtol():
    m0 = MPP(1, 1, rtol=1)
    m1 = MPP(2 + 2 * float_info.epsilon, 2)
    assert m0 != m1
    assert m1 != m0


def test_mpp_match():
    m = match_mpp(MPP(1, 1), MPP(1.1, 1.1), MPP(2.2, 2.2), MPP(3.3, 3.3))
    assert m == MPP(1, 1)

    m = match_mpp(MPP(1, 1, rtol=0.2), MPP(1.1, 1.1), MPP(2.2, 2.2), MPP(3.3, 3.3))
    assert m == MPP(1.1, 1.1)

    m = match_mpp(MPP(1, 1, rtol=0.2), MPP(1.1, 1.1), MPP(2.2, 2.2), MPP(0.95, 0.95))
    assert m == MPP(0.95, 0.95)


def test_mpp_cmp():
    assert MPP(1, 1) > MPP(0.25, 0.25)
    assert not (MPP(1, 1) < MPP(0.25, 0.25))
    assert not (MPP(1, 1) <= MPP(0.25, 0.25))
    assert MPP(1, 1) >= MPP(0.25, 0.25)
