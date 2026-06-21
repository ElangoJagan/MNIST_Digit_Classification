from src.pipeline.train_pipeline import TrainPipeline

pipeline = TrainPipeline()
test_accuracy = pipeline.run_pipeline()
print(f"\n🎯 Final Test Accuracy: {test_accuracy}")