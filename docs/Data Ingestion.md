# Data Ingestion Steps

Step-1. Mention the artifacts for data ingestion in config.yaml

    paths:
        artifacts: artifacts
        data_store: data

Step-2. Specify the name and subset of data we need for NER task in config.yaml

    data_ingestion_config:
        dataset_name: xtreme
        subset_name: PAN-X.en

Step-3. Import all the necessary libraries and also import
```python
from datasets import load_dataset
```

Step-4. Initialize class DataIngestion and connect the directory path 

Step-5. Now we will create a function which will get the data into our data/ directory

```python
def get_data(self):
        try:
            """
            This is class is responsible for data collection from official hugging face library.
            Cross-lingual Transfer Evaluation of Multilingual Encoders (XTREME) benchmark called WikiANN or PAN-X.
            Returns: Dict of train test validation data 
            """
            # Task Implement save data to artifacts/data_store , write check if data already exists there
            # if not then only fetch from load_dataset
            logger.info(f"Loading Data from Hugging face ")

            if not os.path.isdir(self.data_ingestion_config.data_path):
                logger.info(f"No data storage directory found creating directory")
                os.mkdir(self.data_ingestion_config.data_path)

                pan_en_data = load_dataset(self.data_ingestion_config.dataset_name,
                                           name=self.data_ingestion_config.subset_name)
                # saving data locally
                pan_en_data.save_to_disk(self.data_ingestion_config.data_path)

                logger.info(f"Dataset loading completed")
                logger.info(f"Dataset Info : {pan_en_data}")
            else:
                logger.info(f"loading dataset from local disk")
                # reloading dataset from local drive
                pan_en_data = load_from_disk(self.data_ingestion_config.data_path)
                logger.info(f"loading dataset completed")
                return pan_en_data
```

Step-6. The data can be observed as follows

![DataSet directory](./img/cache_dir_data.jpg?raw=true "Dataset directory")