# pado changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...

## [0.10.1] - 2022-09-15
### Fixed
- pado.transporter: switch to platformdirs

## [0.10.0] - 2022-09-14
### Fixed
- pado.images.providers: ensure provided dataframe index is unique
- fix python3.7 compatibility

### Added
- ci: readthedocs
- ci: pypi deployment

## [0.9.0] - 2022-09-08
### Added
- support fuzzy definition of MPP values

### Fixed
- improved checksum comparison

## [0.8.0] - 2022-09-05
### Added
- support freezing datasets

### Fixed
- fixed memory datasets on Windows

## [0.7.0] - 2022-09-02
### Added
- allow disabling urlpath pickling
- support annotation handling in pado.itertools

### Fixed
- support shapely 1.x and 2.x
- various pado.itertools fixes
- resolve high severity bandit issues

## [0.6.1] - 2022-09-01
### Fixed
- pado.itertools: fix TypeError in error handler

## [0.6.0] - 2022-08-31
### Added
- pado.itertools error handler support
- pado.itertools support image storage_options override

### Changed
- move pado dataset creation functionality to pado.create
- pado.images.tiles deprecate old tileiterator
- pado.create support multiprocessing

### Fixed
- improve docs
- fix circular import issues
- fix issue with storage_options passing in find_files

## [0.5.0] - 2022-08-23
### Fixed
- fix in memory dataset pickling
- allow empty metadata providers

### Changed
- improve update_image_provider output
- improve non-image id df provider errors

### Added
- support opening images via other filesystems
- PadoDataset itertools: for use in multiprocessing dataloaders

## [0.4.0] - 2022-08-05
### Added
- started this changelog

[Unreleased]: https://github.com/Bayer-Group/pado/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/Bayer-Group/pado/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/Bayer-Group/pado/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Bayer-Group/pado/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Bayer-Group/pado/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Bayer-Group/pado/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Bayer-Group/pado/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Bayer-Group/pado/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Bayer-Group/pado/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Bayer-Group/pado/tree/v0.4.0
