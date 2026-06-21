from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

obj = DataIngestion()
saved_path = obj.initiate_data_ingestion()

obj_transformation = DataTransformation()
x_train, x_test, y_train_encode, y_test_encode = obj_transformation.initiate_data_transformation(saved_path)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape:  {x_test.shape}")
print(f"y_train_encode shape: {y_train_encode.shape}")
print(f"y_test_encode shape:  {y_test_encode.shape}")
print(f"Sample y_train_encode[0]: {y_train_encode[0]}")