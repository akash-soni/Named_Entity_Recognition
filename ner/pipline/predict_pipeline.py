from ner.config.configurations import Configuration
from ner.exception.exception import CustomException
from ner.components.predictions import PredictionClassifier
from typing import Any, Dict, List
import logging
import sys

logger = logging.getLogger(__name__)


class Prediction_Pipeline:
    def __init__(self, config: object):
        self.config = config

    def run_model_prediction(self):
        try:
            logger.info(" prediction started ")
            prediction_tags = PredictionClassifier(model_prediction_config=self.config.get_model_predict_pipeline_config())
            prediction_tags.prediction()
            logger.info(" Prediction Completed ")

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = Prediction_Pipeline(Configuration())
    pipeline.run_model_prediction()
