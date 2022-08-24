# PADO: PAthological Data Obsession

[![Milestones](https://img.shields.io/badge/mlr%20milestones-pado-brightgreen)](https://github.com/Bayer-Group/pado/milestones?direction=asc&sort=due_date&state=open)

Welcome to `pado` :wave:, a dataset library for accessing histopathological
datasets in a standardized way from [Python](https://www.python.org/).

`pado`'s goal is to provide a unified way to access data from diverse
datasets. Its scope is very small and the design tries to keep everything
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

pado can be installed directly via `pip`:
```bash
pip install "git+https://github.com/Bayer-Group/pado@main#egg=pado[cli,create]"
```

or for development you can clone and install via:
```bash
git clone https://github.com/Bayer-Group/pado.git
cd pathdrive-pado
pip install -e ".[cli,create,dev]"
```

if you prefer conda environments:
```bash
git clone https://github.com/Bayer-Group/pado.git
cd pathdrive-pado
conda install conda-devenv
conda devenv
conda activate pado
```

Note that in this environment `pado` is already installed in development mode,
so go ahead and hack.


## Contributing Guidelines

- Please use [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- When contributing code, please try to use Pull Requests.
- tests go hand in hand with modules on ```tests``` packages at the same level. We use ```pytest```.
- Please install [pre-commit](https://pre-commit.com/) and install the hooks by running `pre-commit install` in the project root folder.

You can setup your IDE to help you adhering to these guidelines.
<br>
_([Santi](https://github.com/sdvillal) is happy to help you setting up pycharm in 5 minutes)_


## Acknowledgements

Build with love by Santi Villalba and Andreas Poehlmann from the _Machine Learning Research_ group at Bayer.

`pado`: copyright 2020-2022 Bayer AG
