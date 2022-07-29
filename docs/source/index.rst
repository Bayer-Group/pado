
.. pado documentation

Welcome to PADO
===============

`pado` (**PA**\ thological **D**\ ata **O**\ bsession) is a
`Python <https://www.python.org/>`_ module which defines a pragmatic
data standard for pathological datasets.

`pado`'s goal is to provide a common way of accessing pathological data.
It does this by enforcing as little structure as necessary to be able to
serve the data via a common interface, while at the same time allowing
for as much freedom as possible to accommodate a variety of use cases.
It's oriented towards Python developers that are familiar with the
numerical Python stack and tries to make pragmatic design choices in
favor of using common tools from the numerical python world to keep the
learning curve flat.

We strive to make your lives as easy as possible: If `pado` is not pythonic,
unintuitive, slow or if its documentation is confusing, it's a bug in
`pado`. Feel free to report any issues or feature requests in the issue
tracker on
`github <https://github.com/Bayer-Group/pado>`_.

This page hosts the documentation for version "|version|".

.. warning::
    Pado is undergoing heavy development right now and things might change
    frequently. Watch the github repository to stay up to date.

.. sidebar:: Acknowledgements

    Build with love by Santi Villalba and Andreas Poehlmann from the
    *Machine Learning Research* group at Bayer. In collaboration with the
    *Pathology Lab 2* and the *Mechanistic and Toxicologic Pathology* group.

.. toctree::
   :maxdepth: 2

   self
   installation
   quickstart

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:

   api/annotations
   api/images
   api/io
   api/metadata
   api/predictions
   api/collections
   api/dataset
   api/mock
   api/settings
   api/shutil
   api/types
