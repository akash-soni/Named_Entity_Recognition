# Data Ingestion Steps

Step-1. Mention the artifacts for data ingestion in config.yaml

    artifacts:
        artifacts_dir: artifacts
        cache_dir: cache_dir

Step-2. Specify the name and subset of data we need for NER task in config.yaml

    dataset:
        name: xtreme
        subset: PAN-X.en

Step-3. Import all the necessary libraries and also import
```python
from datasets import load_dataset
```

Step-4. Initialize class DataIngestion and connect the directory path to cache_dir

Step-5. Now we will create a function which will get the data into our cache_dir/ directory

```python
def get_data(self):
    en_data = load_dataset(self.dataset_name, self.subset, cache_dir = self.cache_dir)
```

Step-6. The data can be observed as follows

![DataSet directory](./img/cache_dir_data.jpg?raw=true "Dataset directory")