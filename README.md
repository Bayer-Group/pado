# PADO: PAthological Data Obsession

[![GitHub issues](https://img.shields.io/github/issues/Bayer-Group/pado)](https://github.com/Bayer-Group/pado/issues)

Welcome to `pado` :wave:, a dataset library for accessing histopathological
datasets in a standardized way from [Python](https://www.python.org/).

`pado`'s goal is to provide a unified way to access data from diverse
datasets. It's scope is very small and the design tries to keep everything
simple.

As always: If `pado` is not pythonic,
unintuitive, slow or if its documentation is confusing, it's a bug in
`pado`. Feel free to report any issues or feature requests in the issue
tracker!

Development
[happens on github](https://github.com/Bayer-Group/pado)
:octocat:

## Documentation

TBA

## Development Installation

1. Install conda and git
2. Clone pado `git clone https://github.com/Bayer-Group/pado.git`
3. Run `conda env create -f environment.yaml`
4. Activate the environment `conda activate pado`

Note that in this environment `pado` is already installed in development mode,
so go ahead and hack.


## Contributing Guidelines

- Please follow [pep-8 conventions](https://www.python.org/dev/peps/pep-0008/) but:
  - We allow 120 character long lines (try anyway to keep them short)
- Please use [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- When contributing code, please try to use Pull Requests.
- tests go hand in hand with modules on ```tests``` packages at the same level. We use ```pytest```.

You can setup your IDE to help you adhering to these guidelines.
<br>
_([Santi](https://github.com/sdvillal) is happy to help you setting up pycharm in 5 minutes)_


## Acknowledgements

Build with love by Santi Villalba and Andreas Poehlmann from the _Machine
Learning Research_ group at Bayer.

`pado`: copyright 2020-2022 Bayer AG
