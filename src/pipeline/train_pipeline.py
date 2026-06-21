
import sys

from src.logger import Logger
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

_logger_obj = Logger("TrainPipeline")
logger = _logger_obj.get_logger()
class TrainPipeline:
    try:
        def __init__(self):
            
            self.data_ingestion = DataIngestion()
            self.data_transformation = DataTransformation()
            self.modeltrain = ModelTrainer()
            
        def run_pipeline(self):
            
            saved_path = self.data_ingestion.initiate_data_ingestion()
            x_train, x_test, y_train_encode, y_test_encode = self.data_transformation.initiate_data_transformation(saved_path)
            test_accuracy = self.modeltrain.initiate_model_trainer(x_train, x_test, y_train_encode, y_test_encode)
            
            return test_accuracy
    except Exception as e:
        raise CustomException(e,sys)