from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import Logger
import sys
import os
import numpy as np

from tensorflow.keras.datasets import mnist

_logger_obj = Logger('Data_ingestion')
logger = _logger_obj.get_logger()

class DataIngestion:

    def __init__(self):
        self.config = DataIngestionConfig()
        
    @staticmethod
    def mnist_load_dataset():
        try:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            return x_train, y_train, x_test, y_test
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def save_data(self, x_train, y_train, x_test, y_test):
        try:
            path = self.config.raw_data_path
            os.makedirs(os.path.dirname(path), exist_ok = True)
            
            np.savez(
                path, 
                x_train= x_train,
                y_train= y_train,
                x_test = x_test,
                y_test = y_test
            )
            return path
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self):
        #1 load data
        try:
            logger.info("=" * 50)
            logger.info("Data Ingestion Started!")
            logger.info("=" * 50)
            
            x_train, y_train, x_test, y_test = self.mnist_load_dataset()
            saved_path = self.save_data(x_train, y_train, x_test, y_test)
            return saved_path
        except Exception as e:
            raise CustomException(e,sys)