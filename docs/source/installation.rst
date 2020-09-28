Installation
============

There's multiple ways to install `pado`. We offer pypi packages and will soon provide
conda packages too. Right now the easiest way to help develop is to setup your environment via conda.


Install pado's dev environment
------------------------------

.. tip::
    This is currently the best way to get started |:snake:|

.. code-block:: console

    user@computer:~$ git clone https://github.com/Bayer-Group/pado.git
    user@computer:~$ cd pado
    user@computer:pado$ conda env create -f environment.yml

This will create a **pado** conda environment with everything you need to get started.
You should now be able to run:

.. code-block:: console

    user@computer:pado$ conda activate pado
    (pado) user@computer:pado$ pytest

And you should see that all the tests pass |:heart:|
