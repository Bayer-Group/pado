"""file utility functions"""
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


def hash_file(path, hasher=hashlib.sha256) -> str:
    """calculate the hash of a file"""
    hasher = hasher()
    with open(path, 'rb') as reader:
        for chunk in iter(lambda: reader.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_zip(path, hasher=hashlib.sha256) -> str:
    """calculate the hash of a zip file

    Uses full paths and file contents to hash a zip file.
    That is, builds a hash based on paths within zip and content of each decompressed file.
    Ignores everything else, for example, file date.
    """
    hasher = hasher()
    with ZipFile(path) as reader:
        for file_info in sorted(reader.infolist(), key=lambda x: x.filename):
            hasher.update(file_info.filename.encode('utf-8'))
            with reader.open(file_info.filename, mode='r') as file_reader:
                for chunk in iter(lambda: file_reader.read(8192), b''):
                    hasher.update(chunk)
    return hasher.hexdigest()


class FilesDupeFinder:

    def __init__(self, hasher=hash_zip, paths=None):
        """find duplicate files according to hash implementation"""
        super().__init__()
        self.dupes = defaultdict(set)  # a map hash -> files
        self.hasher = hasher
        self.update(*(paths or ()))

    def update(self, *paths):
        for path in paths:
            if Path(path).is_file():
                self.dupes[self.hasher(path)].add(str(path))

    def dupes(self):
        return sorted(group for group in self.dupes.values() if len(group) > 1)


def zip_inplace(dest_dir, path, delete_path=False, file_format='zip'):
    """create an archive of the directory at `path` and store at dest_dir"""
    dest_dir, path = Path(dest_dir), Path(path)
    if not path.is_dir():
        raise ValueError('path {} must be a directory'.format(path))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = dest_dir / path.stem
    shutil.make_archive(base_name=dest_zip,
                        format=file_format,
                        root_dir=path.parent,
                        base_dir=path.name)
    if delete_path:
        shutil.rmtree(path)
