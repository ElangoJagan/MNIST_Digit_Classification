from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path = 'data/raw/mnist_data.npz'
    
class DataTransformationConfig:
    image_height = 28
    image_width = 28
    num_classes = 10
    
@dataclass
class ModelTrainerConfig:
    #Hyperparameters
    model_path= 'artifacts/model.h5'
    epochs = 10
    batch_size = 32
    num_classes = 10
    