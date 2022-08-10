from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_name", "subset_name", "data_path"])

DataValidationConfig = namedtuple("DataValidationConfig", 
["dataset", "data_split", "columns_check", "type_check", "null_check"])