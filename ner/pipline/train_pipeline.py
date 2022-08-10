from ner.config.configurations import Configuration
from ner.components.data_ingestion import DataIngestion
from ner.exception.exception import CustomException
from typing import Any, Dict, List
import logging
import sys

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: object):
        self.config = config

    def run_data_ingestion(self) -> Dict:
        try:
            logger.info(" Running Data Ingestion pipeline ")
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            data = data_ingestion.get_data()
            return data
        except Exception as e:
            raise CustomException(e, sys)


    def run_pipeline(self):
        data = self.run_data_ingestion()


if __name__ == "__main__":
    pipeline = Pipeline(Configuration())
    pipeline.run_pipeline()
