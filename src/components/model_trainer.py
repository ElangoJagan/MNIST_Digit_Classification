import sys
import os
import numpy as np

from src.exception import CustomException
from src.logger import Logger
from src.entity.config_entity import ModelTrainerConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense


_logger_obj = Logger('Model_trainer')
logger = _logger_obj.get_logger()
class ModelTrainer:
    
    def __init__(self):
        self.config = ModelTrainerConfig()
        
    def build_models(self):
        try:
            model = Sequential([
                Flatten(input_shape=(28,28)),
                Dense(128, activation= 'relu'),
                Dense(self.config.num_classes, activation= 'softmax')
            ])
            model.compile(
                optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy']
                
            )
            return model
        except Exception as e:
            raise CustomException(e,sys)
    def train_model(self, model, x_train, y_train):
        try:
            model.fit(
                x_train, y_train, epochs = self.config.epochs, batch_size = self.config.batch_size
            )
            return model
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_model_trainer(self, x_train, x_test, y_train_encode, y_test_encode):
        try:
            #build model 
            model = self.build_models()
            
            #train model
            model = self.train_model(model, x_train, y_train_encode)
            
            #evaluate model on test
            test_loss, test_accuracy = model.evaluate(x_test, y_test_encode)
            
            
            #save the model 
            model.save(self.config.model_path)
            
            return test_accuracy
        except Exception as e:
            raise CustomException(e,sys)
        