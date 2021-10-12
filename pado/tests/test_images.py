import os
import pickle

import pytest

from pado.images import ImageId


# --- test constructors -----------------------------------------------

IMAGE_ID_ARG_KWARG_LIST = [
    pytest.param([("a",), {}], id="single_part"),
    pytest.param([("a", "b", "c"), {}], id="multi_part"),
    pytest.param([("a b",), {}], id="part_with_space"),
    pytest.param([("a\u1234b",), {}], id="part_with_unicode"),
    pytest.param([("a", "b"), {"site": "mars"}], id="with_site"),
]


@pytest.mark.parametrize(
    "image_id_args_kwargs", IMAGE_ID_ARG_KWARG_LIST
)
def test_image_id___new__(image_id_args_kwargs):
    args, kwargs = image_id_args_kwargs
    _ = ImageId(*args, **kwargs)  # just test the constructor


@pytest.mark.parametrize(
    "image_id_args_kwargs", [
        pytest.param([(), {}], id="empty"),
        pytest.param([(1,), {}], id="non_string_only"),
        pytest.param([("a", 1), {}], id="non_string_some"),
        pytest.param([(["a", "b"],), {}], id="iterable_at_args0"),
        pytest.param([("ImageId('a')",), {}], id="str_serialized_image_id"),
        pytest.param([('{"image_id":["a"]}',), {}], id="json_serialized_image_id"),
    ]
)
def test_image_id_incorrect_input(image_id_args_kwargs):
    args, kwargs = image_id_args_kwargs
    with pytest.raises(ValueError):
        ImageId(*args, **kwargs)


@pytest.mark.parametrize(
    "image_id_arg_site", [
        pytest.param(["a", None], id="str"),
    ]
)
def test_image_id_make_incorrect_input(image_id_arg_site):
    arg, site = image_id_arg_site
    with pytest.raises(TypeError):
        ImageId.make(arg, site=site)


def test_image_id_make():
    iid = ImageId.make(['a', 'b'], site="mars")
    assert iid == ImageId('a', 'b', site="mars")


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
    new_id = pickle.loads(serialized)
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
    "id_input", [
        pytest.param("", id="empty_str"),
        pytest.param("('a')", id="tuple_str"),
        pytest.param("ImageId('a'')", id="unparsable"),
        pytest.param('{"image_id":["a"]}', id="json"),
    ]
)
def test_image_id_from_str_incorrect_input(id_input):
    with pytest.raises(ValueError):
        _ = ImageId.from_str(id_input)


def test_image_id_from_str_with_subclass():
    # note: this is basically irrelevant... very unlikely anyone
    #   will subclass this ever. we could mark the class `final`
    #   but... *screams in questionable new python features*
    serialized = ImageId('a', 'b', site='mars').to_str()

    class Why(ImageId):
        pass

    with pytest.raises(ValueError):
        Why.from_str(serialized)


@pytest.mark.parametrize("image_id", IMAGE_ID_ARG_KWARG_LIST, indirect=True)
def test_image_id_json_roundtrip(image_id):
    id_json = image_id.to_json()
    new_id = ImageId.from_json(id_json)
    assert image_id == new_id


def test_image_id_from_json_incorrect_input_type():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ImageId.from_json(1)


@pytest.mark.parametrize(
    "id_input", [
        pytest.param("", id="empty_str"),
        pytest.param("ImageId('a')", id="str_id"),
        pytest.param('{"image_id":"filename.svs"}', id="incorrect_json"),
    ]
)
def test_image_id_from_json_incorrect_input(id_input):
    with pytest.raises(ValueError):
        _ = ImageId.from_json(id_input)


# --- test equality and hashing ---------------------------------------

def test_image_id_eq_dont_coerce():
    iid = ImageId('a', 'b', site='mars')
    iid_tuple = ('mars', 'a', 'b')
    assert (iid == iid_tuple) is False
    assert iid != iid_tuple


def test_image_id_to_base64_encoding():
    iid = ImageId('a', 'b', site='mars')
    iid_b64_encoded = "SW1hZ2VJZCgnYScsICdiJywgc2l0ZT0nbWFycycp"
    assert iid.to_url_id() == iid_b64_encoded


def test_image_id_current_hash_assumption():
    iid = ImageId('a', 'b', site='mars')
    fn_tuple = ('b',)
    assert hash(iid) == hash(fn_tuple)
    assert iid != fn_tuple


def test_image_id_without_site_when_used_as_key():
    iid_with_site = ImageId('a', 'b', site='mars')
    iid_no_site = ImageId('a', 'b')
    wrong_iid_no_site = ImageId('a', 'c')

    data = {iid_with_site: "metadata"}
    assert data.get(wrong_iid_no_site, None) is None
    assert data.get(iid_no_site, None) == "metadata"


# --- test path handling ----------------------------------------------

def test_image_id_site_when_mapper_not_available():
    iid = ImageId('a', 'b', site='site-does-not-exist')

    with pytest.raises(KeyError):
        _ = iid.id_field_names
    with pytest.raises(KeyError):
        _ = os.fspath(iid)
    with pytest.raises(KeyError):
        _ = iid.to_path()


def test_image_id_file_mapper_id_field_names():
    iid = ImageId('a', 'b')  # no site!
    assert iid.id_field_names == ('site', 'filename')


def test_image_id_file_mapper_fspath():
    iid = ImageId('a', 'b')  # no site!
    assert os.fspath(iid)


def test_image_id_file_mapper_to_path():
    iid = ImageId('a', 'b')  # no site!
    assert iid.to_path()


# --- test in datasets ------------------------------------------------

def test_image_id_in_dataset(dataset_ro):
    key = next(iter(dataset_ro.images))
    assert isinstance(key, ImageId)
    assert key in dataset_ro.annotations
    assert key in dataset_ro.metadata
