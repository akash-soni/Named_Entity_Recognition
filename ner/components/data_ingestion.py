import os.path
from ner.config.configurations import Configuration
from ner.entity.config_entity import DataIngestionConfig
from ner.exception.exception import CustomException
from datasets import load_dataset, load_from_disk


import logging
import sys

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        logger.info(" Data Ingestion Log Started ")
        self.data_ingestion_config = data_ingestion_config
        print(data_ingestion_config.data_path)

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


        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)


if __name__ == "__main__":
    project_config = Configuration()
    ingestion = DataIngestion(project_config.get_data_ingestion_config())
    print(ingestion)
