import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline

# Load a sample test image from MNIST to verify prediction works
data = np.load("data/raw/mnist_data.npz")
sample_image = data['x_test'][0]    # take the FIRST test image
actual_label = data['y_test'][0]     # the REAL correct answer

pipeline = PredictPipeline()
predicted_digit = pipeline.predict(sample_image)

print(f"\n🔮 Predicted Digit: {predicted_digit}")
print(f"✅ Actual Digit:    {actual_label}")