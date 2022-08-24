Quickstart
==========

Here are some quickstart examples for how to use pado.

Playing with an example dataset
-------------------------------

If you need a super quick example of how to use a pado dataset, you can create a
fake dataset, that's also used in the internal tests.

.. code-block:: python

    >>> from pado.mock import mock_dataset
    >>> ds = mock_dataset(None)
    >>> ds
    PadoDataset('memory://pado-f5869e41-5246-4378-9057-96fda1c40edf', mode='r+')

This creates a test dataset in memory with 3 images and some fake metadata

.. code-block:: python

    >>> len(ds)
    3
    >>> ds.index
    (ImageId('mock_image_0.svs', site='mock'),
     ImageId('mock_image_1.svs', site='mock'),
     ImageId('mock_image_2.svs', site='mock'))
    >>> ds[0].image
    Image(...)
    >>> ds[0].metadata
                                              A  B  C  D
    ImageId('mock_image_0.svs', site='mock')  a  2  c  4


Creating a pado dataset
-----------------------

We're soon adding an example of how to create a pado dataset.

Using a pado dataset remotely
-----------------------------

We're soon adding an example of how to use the remote capabilities of a pado dataset.

Using a pado dataset with pytorch
---------------------------------

`pado` datasets provide tools to be used in dataloaders.

.. code-block:: python

    from pado import PadoDataset
    from pado.itertools import TileDataset
    from pado.images.tiles import FastGridTiling
    from pado.images.utils import MPP

    ds = PadoDataset("/path/to/my/dataset", mode="r")

    # map-style dataset only accessing tissue tiles
    dataset = TileDataset(
        ds,
        tiling_strategy=FastGridTiling(
            tile_size=(512, 512),
            target_mpp=MPP(1, 1),
            overlap=100,
            min_chunk_size=0.2,
            normalize_chunk_sizes=True,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )


More examples
^^^^^^^^^^^^^

We need your input! |:bow:|

.. tip::
    In case you need another example for the specific thing you'd like to do, please feel free to open a new
    issue on `pado`'s `issue tracker <https://github.com/Bayer-Group/pado/issues>`_.
    We'll try our best to help you |:+1:|
