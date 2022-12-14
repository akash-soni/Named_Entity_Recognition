from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_name", "subset_name", "data_path"])

DataValidationConfig = namedtuple("DataValidationConfig", 
["dataset", "data_split", "columns_check", "type_check", "null_check"])

DataPreprocessingConfig = namedtuple("DataPreprocessingConfig", ["model_name", "tags", "index2tag",
                                                                 "tag2index", "tokenizer"])

ModelTrainConfig = namedtuple("ModelTrainConfig", ["model_name", "index2tag", "tag2index",
                                                   "tokenizer", "xlmr_config", "epochs",
                                                   "batch_size", "save_steps", "output_dir"])

ModelPredConfig = namedtuple("ModelPredConfig", ["model_name", "index2tag", "tag2index",
                                                   "tokenizer", "xlmr_config", "model_dir"])

TestPipelineConfig = namedtuple("TestPipelineConfig", ["dataset_name", "subset_name"])