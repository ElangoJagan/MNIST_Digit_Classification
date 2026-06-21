import sys
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model

from src.entity.config_entity import ModelTrainerConfig
from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger("PredictPipeline")
logger = _logger_obj.get_logger()


class BasePipeline(ABC):
    """
    Abstract Base Class — defines the CONTRACT.
    Just the RULE, no actual implementation here!
    """
    @abstractmethod
    def predict(self, input_data):
        pass


class PredictPipeline(BasePipeline):
    """
    REAL implementation — loads model, makes predictions.
    """
    
    def __init__(self):
        try:
            self.config = ModelTrainerConfig()
            model_path = self.config.model_path
            logger.info(f"Loading model from {model_path}")
            self.model = load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Error loading model")
            raise CustomException(e, sys)
    
    def predict(self, input_img):
        try:
            logger.info("Prediction started")
            
            # Step 1 — normalize (FIXED: now assigned!)
            input_img = input_img / 255
            
            # Step 2 — reshape to add batch dimension
            input_reshaped = input_img.reshape(1, 28, 28)
            
            # Step 3 — predict
            prediction = self.model.predict(input_reshaped)
            
            # Step 4 — get digit with highest probability
            predicted_digit = np.argmax(prediction)
            
            logger.info(f"Predicted digit: {predicted_digit}")
            return predicted_digit
            
        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)