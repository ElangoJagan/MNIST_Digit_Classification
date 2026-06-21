import sys
from flask import Flask, render_template, request
from PIL import Image
import numpy as np


from src.logger import Logger
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline


_logger_obj = Logger('app')
logger = _logger_obj.get_logger()



app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files.get("image")
        # 2 — open and process image
        image = Image.open(uploaded_file)
        image = image.convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image)

        # 🔍 DEBUG CODE — ADD THESE 3 LINES HERE!
        from PIL import Image as PILImage
        debug_img = PILImage.fromarray(image_array)
        debug_img.save("debug_uploaded_image.png")
        print(f"Image array min: {image_array.min()}, max: {image_array.max()}")
        # 🔍 END DEBUG CODE

        logger.info(f"Image processed, shape: {image_array.shape}")

        # Step 3 — predict
        pipeline = PredictPipeline()
        predicted_digit = pipeline.predict(image_array)

        logger.info(f"Predicted digit: {predicted_digit}")

        return render_template("result.html", prediction=int(predicted_digit))

    except Exception as e:
        logger.error("Prediction request failed")
        raise CustomException(e, sys)

if __name__ =='__main__':
    app.run(debug= True)