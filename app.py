from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gdown
from keras.layers import BatchNormalization

app = Flask(__name__)

# Google Drive file ID for the model
FILE_ID = "1cXfXTa0RwLkopFVS2TXv3BpD7eypRYvK"
MODEL_PATH = "defect_model.h5"

# Function to download the model if not found
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading defect_model.h5 from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Download the model if necessary
download_model()

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

# Custom BatchNormalization to fix loading issues
class CustomBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs):
        if isinstance(kwargs.get("axis"), list):
            kwargs["axis"] = kwargs["axis"][0]  # Convert list to int
        super().__init__(*args, **kwargs)

# Load the model with custom objects
try:
    model_resnet = tf.keras.models.load_model(MODEL_PATH, custom_objects={"BatchNormalization": CustomBatchNormalization})
except Exception as e:
    print(f"Error loading model: {e}")
    model_resnet = None  # Set model to None if it fails to load

# Function to preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model_resnet is None:
        return jsonify({'error': 'Model failed to load. Check logs for details.'})

    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'})

        # Save file temporarily
        file_path = 'temp_image.jpg'
        file.save(file_path)

        # Preprocess and predict
        img = preprocess_image(file_path)
        predicted_probabilities = model_resnet.predict(img)

        # Extract prediction results
        predicted_probability = float(predicted_probabilities[0][0])
        predicted_label = "Faulty Casting mold" if predicted_probability >= 0.5 else "Casting mold is good"

        result = {
            'predicted_probability': predicted_probability,
            'predicted_label': predicted_label
        }

        # Remove temporary image file
        os.remove(file_path)

        return render_template('index.html', prediction=result)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)
