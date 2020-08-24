import pathlib


PADO_DATASET_SCHEMA = {
    "pado": {
        "schema": "0.1",
        "type": "dataset",
    }
}


def write_schema(path: pathlib.Path):
    """write the pado dataset schema to a toml file"""
    # TODO:
    #  - this could hold some more information that is dataset wide
    #  - even if we don't extend it, it's good to have versioning
    #    available before we do the first release


