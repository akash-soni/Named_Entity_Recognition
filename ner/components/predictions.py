from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.entity.config_entity import ModelPredConfig
from typing import Any, Dict, AnyStr
import logging
import torch
import sys
import numpy as np

logger = logging.getLogger(__name__)


class PredictionClassifier:

    def __init__(self, model_prediction_config: ModelPredConfig, text: str):
        self.model_prediction_config = model_prediction_config
        self.text = text
    

    def get_pred_ids(self, predictions, word_ids):
        prediction = [i.item() for i in predictions[0]]
        previous_word_idx = None
        pred_ids = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            elif word_idx != previous_word_idx:
                pred_ids.append(prediction[idx])
            previous_word_idx = word_idx

        return pred_ids


    def prediction(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xlmr_fine_model = (
            XLMRobertaForTokenClassification.from_pretrained(self.model_prediction_config.model_dir,
                                                             config = self.model_prediction_config.xlmr_config).to(device)
        )
        #text = input("ENTER TEXT SENTENCE")
        tokenized_input = self.model_prediction_config.tokenizer(self.text.split(), is_split_into_words=True)
        word_ids = tokenized_input.word_ids()

        print(device)
        # convert input_ids into tokens
        data = torch.tensor(tokenized_input['input_ids'])
        data = data.reshape(1, -1)

        outputs = xlmr_fine_model(data.to(device)).logits
        predictions = torch.argmax(outputs, dim=-1)

        prediction_ids = self.get_pred_ids(predictions, word_ids)
        predicted_output = [self.model_prediction_config.index2tag[idx] for idx in prediction_ids]
        # print(predicted_output)
        return predicted_output


