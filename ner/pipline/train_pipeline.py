from ner.config.configurations import Configuration
from ner.components.data_ingestion import DataIngestion
from ner.components.data_validation import DataValidation
from ner.components.data_preperation import DataPreprocessing
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


    def run_data_validation(self, data) -> List[List[bool]]:
        try:
            logger.info(" Running Data validation Pipeline ")
            validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                        data=data)
            checks = validation.drive_checks()
            return checks
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def run_data_preparation(self, data) -> Dict:
        try:
            logger.info(" Running Data Preparation pipeline ")
            data_preprocessing = DataPreprocessing(data_preprocessing_config=self.config.get_data_preprocessing_config(),
                                                  data=data)
            data = data_preprocessing.prepare_data_for_fine_tuning()
            return data
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


    def run_pipeline(self):
        data = self.run_data_ingestion()
        checks = self.run_data_validation(data=data)
        if sum(checks[0]) == 3:
            logger.info("Checks Completed")
            processed_data = self.run_data_preparation(data=data)
        else:
            logger.error("Checks Failed")


if __name__ == "__main__":
    pipeline = Pipeline(Configuration())
    pipeline.run_pipeline()
