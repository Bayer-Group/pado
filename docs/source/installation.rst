Installation
============

There's multiple ways to install `pado`. We will soon offer pypi packages and will soon provide
conda packages too. Right now the easiest way to help develop is to setup your as described below:


Install pado's dev environment
------------------------------

.. tip::
    This is currently the best way to get started |:snake:|

`pado` can be installed directly via `pip`:

.. code-block:: console

    user@computer:~$ pip install "git+https://github.com/Bayer-Group/pado@main#egg=pado[cli,create]"

or for development you can clone and install via:

.. code-block:: console

    user@computer:~$ git clone https://github.com/Bayer-Group/pado.git
    user@computer:~$ cd pathdrive-pado
    user@computer:pathdrive-pado$ pip install -e ".[cli,create,dev]"

if you prefer conda environments:

.. code-block:: console

    user@computer:~$ git clone https://github.com/Bayer-Group/pado.git
    user@computer:~$ cd pathdrive-pado
    user@computer:pathdrive-pado$ conda install conda-devenv
    user@computer:pathdrive-pado$ conda devenv
    user@computer:pathdrive-pado$ conda activate pado

Note that in this environment `pado` is already installed in development mode,
so go ahead and hack.

.. code-block:: console

    (pado) user@computer:pathdrive-pado$ pytest

And you should see that all the tests pass |:heart:|
