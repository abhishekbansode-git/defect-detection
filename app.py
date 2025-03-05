from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gdown  # Import gdown for downloading from Google Drive

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

# Load the pre-trained model
model_resnet = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']

        # Save the file temporarily
        file_path = 'temp_image.jpg'
        file.save(file_path)

        # Preprocess the image
        img = preprocess_image(file_path)

        # Make predictions
        predicted_probabilities = model_resnet.predict(img)

        # Assuming a threshold of 0.5 for binary classification
        threshold = 0.5
        predicted_probability = 0 if predicted_probabilities[0][0] < threshold else 1
        predicted_label = "Faulty Casting mold" if predicted_probabilities[0][0] >= threshold else "Casting mold is good"

        result = {
            'predicted_probability': predicted_probability,
            'predicted_label': predicted_label
        }

        # Remove the temporary file
        os.remove(file_path)

        return render_template('index.html', prediction=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)