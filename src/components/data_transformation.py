import os
import sys
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import Logger
import numpy as np
from tensorflow.keras.utils import to_categorical

_logger_obj = Logger('DataTransformation')
logger = _logger_obj.get_logger()

class DataTransformation:
    
    def __init__(self):
        self.config = DataTransformationConfig()
        
    
    def normalize_data(self, x_train, x_test):
        #  pixel value {0 to 255} so its difficult to converge also form weights aggressively so / by 255
        # to make value between 0 to 1
        try:
            x_train = x_train/255
            x_test = x_test/255
            return x_train, x_test
        except Exception as e:
            raise CustomException(e,sys)
    
    def encode_labels(self, y_train, y_test):
        # whre value mades to 1 and others 0
        try:
            y_train_encode = to_categorical(y_train, num_classes = 10)
            y_test_encode = to_categorical(y_test, num_classes = 10)
            return y_train_encode, y_test_encode
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, raw_data_path):
        try:
            logger.info('DataTransformation started')
            data = np.load(raw_data_path)
            x_train = data['x_train']
            y_train= data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
            
            #normalizee
            x_train, x_test = self.normalize_data(x_train, x_test)
            
            #encode 
            y_train_encode, y_test_encode = self.encode_labels(y_train, y_test)
            
            logger.info('============DataTransformation Completed==========')
            return x_train, x_test, y_train_encode, y_test_encode
        
        except Exception as e:
            raise CustomException(e,sys)
        
            
            
        
        
        
        
        
        