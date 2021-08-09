"""pado.meta.store

provides a single file parquet store for pd.DataFrames with per store metadata
"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from pado.util.store import Store
from pado.util.store import StoreType


__all__ = [
    "MetadataStore",
]


# === storage =================================================================

class MetadataStore(Store):
    """stores the metadata in a single file with per store metadata"""
    METADATA_KEY_DATASET_VERSION = "dataset_version"
    DATASET_VERSION = 1

    def __init__(self):
        super().__init__(version=1, store_type=StoreType.METADATA)

    def __metadata_set_hook__(self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]) -> None:
        setter(dct, self.METADATA_KEY_DATASET_VERSION, self.DATASET_VERSION)

    def __metadata_get_hook__(self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]) -> Optional[dict]:
        dataset_version = getter(dct, self.METADATA_KEY_DATASET_VERSION, None)
        if dataset_version is None or dataset_version < self.DATASET_VERSION:
            raise RuntimeError("Please migrate MetadataStore to newer version.")
        elif dataset_version > self.DATASET_VERSION:
            raise RuntimeError("MetadataStore is newer. Please upgrade pado to newer version.")
        return {
            self.METADATA_KEY_DATASET_VERSION: dataset_version
        }
