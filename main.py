from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

obj = DataIngestion()
saved_path = obj.initiate_data_ingestion()

obj_transformation = DataTransformation()
x_train, x_test, y_train_encode, y_test_encode = obj_transformation.initiate_data_transformation(saved_path)

obj_modeltrainer = ModelTrainer()
test_accuracy = obj_modeltrainer.initiate_model_trainer(x_train, x_test, y_train_encode, y_test_encode)
