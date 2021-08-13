# Dev notes:

Quickly make a dataset for playing with a pado dataset:

```python
>>> ds = PadoDataset("/myfolder/padotestdataset", mode="a")
>>> testsource = TestDataSource(num_images=5, num_findings=20, identifier="mytestsource")
>>> ds.add_source(test_datasource)
# will write a pado dataset to disk
```

```python
from pado.dataset import PadoDataset

ds = PadoDataset("/myfolder/padotestdataset", mode="r")
```