from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
import gdown

app = Flask(__name__)
CORS(app)

# Create models directory if it doesn't exist
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file IDs (replace with your actual IDs)
BRAIN_CLASSIFICATION_MODEL_ID = "https://drive.google.com/file/d/1NbmXviD0dzBck5Kfiw4Ybt24Kqt1Jjmv/view?usp=drive_link"
BRAIN_SEGMENTATION_MODEL_ID = "https://drive.google.com/file/d/1mMaidey49WG1Kk4Evq2RFTTqXsKZxe2q/view?usp=drive_link"
SKIN_MODEL_ID = "https://drive.google.com/file/d/1HAcF3evhH8A4V_EnCWTVN0Gw9iU0VxnI/view?usp=drive_link"

# Paths to save the downloaded model files
BRAIN_CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "Brain_Tumor_Classification_Model.h5")
BRAIN_SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_segmentation_model.h5")
SKIN_MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_model.h5")

# Function to download model from Google Drive
def download_model_if_not_exists(drive_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, output_path, quiet=False)

# Download models if not present
download_model_if_not_exists(BRAIN_CLASSIFICATION_MODEL_ID, BRAIN_CLASSIFICATION_MODEL_PATH)
download_model_if_not_exists(BRAIN_SEGMENTATION_MODEL_ID, BRAIN_SEGMENTATION_MODEL_PATH)
download_model_if_not_exists(SKIN_MODEL_ID, SKIN_MODEL_PATH)

# Load models
try:
    brain_classification_model = keras.models.load_model(BRAIN_CLASSIFICATION_MODEL_PATH)
    brain_segmentation_model = keras.models.load_model(BRAIN_SEGMENTATION_MODEL_PATH)
    skin_model = keras.models.load_model(SKIN_MODEL_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    brain_classification_model = None
    brain_segmentation_model = None
    skin_model = None

# Preprocessing for brain tumor models (128x128)
def preprocess_brain_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Preprocessing for skin cancer model (28x28)
def preprocess_skin_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/scan", methods=["POST"])
def scan_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if "diseaseType" not in request.form:
        return jsonify({"error": "No disease type provided"}), 400

    image_file = request.files["image"]
    disease_type = request.form["diseaseType"]
    image_bytes = image_file.read()
    results = {}

    if disease_type == "Brain Tumor" and brain_classification_model and brain_segmentation_model:
        try:
            processed_image = preprocess_brain_image(image_bytes)
            segmentation_output = brain_segmentation_model.predict(processed_image)
            segmentation_mask = (segmentation_output[0, :, :, 0] * 255).astype(np.uint8)
            seg_image = Image.fromarray(segmentation_mask, 'L')
            buffered = io.BytesIO()
            seg_image.save(buffered, format="PNG")
            encoded_seg_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            results["segmentation_image_base64"] = encoded_seg_image

            prediction = brain_classification_model.predict(processed_image)
            class_names = ["Glioma", "Meningioma", "No tumor", "Pituitary tumor"]
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            results["diagnosis"] = predicted_class_name
            results["confidence"] = f"{confidence:.2f}%"

            if predicted_class_name == "No tumor":
                results["description"] = "No brain tumor was detected."
                results["recommendations"] = ["Routine check-up", "Healthy lifestyle"]
            else:
                results["description"] = f"{predicted_class_name} likely present. Seek medical advice."
                results["recommendations"] = [
                    "Consult specialist", "MRI with contrast", "Biopsy if needed", "Discuss treatment plan"
                ]

            results["segmentation_info"] = "Segmentation completed."

        except Exception as e:
            return jsonify({"error": f"Brain tumor analysis failed: {e}"}), 500

    elif disease_type == "Skin Cancer" and skin_model:
        try:
            processed_image = preprocess_skin_image(image_bytes)
            prediction = skin_model.predict(processed_image)
            class_names = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions",
                           "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"]
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            results["diagnosis"] = predicted_class_name
            results["confidence"] = f"{confidence:.2f}%"

            if predicted_class_name in ["Melanoma", "Basal cell carcinoma"]:
                results["description"] = f"{predicted_class_name} detected. Requires medical attention."
                results["recommendations"] = [
                    "See dermatologist", "Biopsy recommended", "Sun protection", "Regular check-ups"
                ]
            else:
                results["description"] = f"{predicted_class_name} appears benign."
                results["recommendations"] = [
                    "Monitor for changes", "Use sunscreen", "Consult if suspicious changes occur"
                ]
        except Exception as e:
            return jsonify({"error": f"Skin cancer analysis failed: {e}"}), 500

    else:
        return jsonify({"error": "Invalid disease type or model not loaded"}), 400

    return jsonify(results)

@app.route("/health", methods=["GET"])
def health_check():
    model_status = {
        "brain_classification": brain_classification_model is not None,
        "brain_segmentation": brain_segmentation_model is not None,
        "skin_cancer": skin_model is not None
    }
    return jsonify({
        "status": "healthy",
        "models_loaded": model_status,
        "message": "Medical AI API is running"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
