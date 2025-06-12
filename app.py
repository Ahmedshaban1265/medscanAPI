import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS to allow React app to connect

# Paths to model files
MODEL_DIR = "./models"
BRAIN_CLASSIFICATION_MODEL_NAME = "Brain_Tumor_Classification_Model.h5"
BRAIN_SEGMENTATION_MODEL_NAME = "brain_tumor_segmentation_model.h5"
SKIN_MODEL_NAME = "skin_cancer_model.h5"

BRAIN_CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, BRAIN_CLASSIFICATION_MODEL_NAME)
BRAIN_SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, BRAIN_SEGMENTATION_MODEL_NAME)
SKIN_MODEL_PATH = os.path.join(MODEL_DIR, SKIN_MODEL_NAME)

# Google Drive direct download links
# Replace these with your actual direct download links
BRAIN_CLASSIFICATION_MODEL_URL = "https://drive.google.com/uc?export=download&id=1HAcF3evhH8A4V_EnCWTVN0Gw9iU0VxnI"
BRAIN_SEGMENTATION_MODEL_URL = "https://drive.google.com/uc?export=download&id=1mMaidey49WG1Kk4Evq2RFTTqXsKZxe2q"
SKIN_MODEL_URL = "https://drive.google.com/uc?export=download&id=1NbmXviD0dzBck5Kfiw4Ybt24Kqt1Jjmv"

def download_file(url, filename):
    """Downloads a file from a given URL."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return False
    else:
        print(f"{filename} already exists. Skipping download.")
    return True

# Download models if they don't exist
print("Checking for AI models...")
download_file(BRAIN_CLASSIFICATION_MODEL_URL, BRAIN_CLASSIFICATION_MODEL_NAME)
download_file(BRAIN_SEGMENTATION_MODEL_URL, BRAIN_SEGMENTATION_MODEL_NAME)
download_file(SKIN_MODEL_URL, SKIN_MODEL_NAME)

# Load the models when the application starts
try:
    # Note: The brain classification model actually expects 28x28x3 input, not 128x128x3
    # This was discovered by examining the model configuration
    brain_classification_model = keras.models.load_model(BRAIN_CLASSIFICATION_MODEL_PATH, compile=False)
    brain_segmentation_model = keras.models.load_model(BRAIN_SEGMENTATION_MODEL_PATH, compile=False)
    skin_model = keras.models.load_model(SKIN_MODEL_PATH, compile=False)
    print("AI models loaded successfully!")
    print(f"Brain classification model input shape: {brain_classification_model.input_shape}")
    print(f"Brain segmentation model input shape: {brain_segmentation_model.input_shape}")
    print(f"Skin cancer model input shape: {skin_model.input_shape}")
except Exception as e:
    print(f"Error loading AI models: {e}")
    brain_classification_model = None
    brain_segmentation_model = None
    skin_model = None

# Function to preprocess the image for brain classification model (28x28)
def preprocess_brain_classification_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 128x128 for brain classification model (corrected size)
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values to 0-1 range
    
    # Ensure 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4:  # Remove alpha channel if present
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to preprocess the image for brain segmentation model (128x128)
def preprocess_brain_segmentation_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 128x128 for brain segmentation model
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values to 0-1 range
    
    # Ensure 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4:  # Remove alpha channel if present
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to preprocess the image for skin model (128x128) - CORRECTED SIZE
def preprocess_skin_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 128x128 for skin model (corrected size)
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values to 0-1 range
    
    # Ensure 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4:  # Remove alpha channel if present
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
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
            # Step 1: Segmentation (uses 128x128 input)
            processed_image_seg = preprocess_brain_segmentation_image(image_bytes)
            segmentation_output = brain_segmentation_model.predict(processed_image_seg)
            
            # Process segmentation output to an image and encode as Base64
            # Assuming segmentation_output is a mask with values between 0 and 1
            segmentation_mask = (segmentation_output[0, :, :, 0] * 255).astype(np.uint8)
            seg_image = Image.fromarray(segmentation_mask, 'L')  # 'L' for grayscale
            
            buffered = io.BytesIO()
            seg_image.save(buffered, format="PNG")
            encoded_seg_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            results["segmentation_image_base64"] = encoded_seg_image
            
            # Step 2: Classification (uses 28x28 input)
            processed_image_class = preprocess_brain_classification_image(image_bytes)
            prediction = brain_classification_model.predict(processed_image_class)
            print(f"Brain classification prediction shape: {prediction.shape}")
            print(f"Brain classification prediction: {prediction}")
            
            # Brain classification model outputs probabilities for the 4 classes:
            # [Glioma, Meningioma, Pituitary tumor, No tumor]
            class_names = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            results["diagnosis"] = predicted_class_name
            results["confidence"] = f"{confidence:.2f}%"
            
            if predicted_class_name == "No tumor":
                results["description"] = "Based on the AI analysis, no brain tumor was detected. Regular check-ups are recommended."
                results["recommendations"] = ["Routine check-up", "Maintain healthy lifestyle"]
            else:
                results["description"] = f"Based on the AI analysis, a {predicted_class_name} is likely present. This diagnosis is based on the classification model after image preprocessing and segmentation. Further medical consultation and detailed imaging are highly recommended for confirmation and treatment planning."
                results["recommendations"] = ["Consult a neurologist/oncologist", "Further specialized imaging (e.g., MRI with contrast)", "Biopsy for definitive diagnosis", "Discuss treatment options with your healthcare provider"]
            
            results["segmentation_info"] = "Brain segmentation model processed and mask generated."
            
        except Exception as e:
            return jsonify({"error": f"Brain tumor analysis failed: {e}"}), 500

    elif disease_type == "Skin Cancer" and skin_model:
        try:
            # Preprocess image for skin model (128x128) - CORRECTED
            processed_image = preprocess_skin_image(image_bytes)
            
            prediction = skin_model.predict(processed_image)
            
            # Assuming the skin cancer model outputs probabilities for 7 classes
            # Based on HAM10000 dataset: [AKIEC, BCC, BKL, DF, MEL, NV, VASC]
            class_names = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions", 
                          "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"]
            
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100
            
            results["diagnosis"] = predicted_class_name
            results["confidence"] = f"{confidence:.2f}%"
            
            # Determine if malignant or benign
            malignant_classes = ["Melanoma", "Basal cell carcinoma"]
            
            if predicted_class_name in malignant_classes:
                results["description"] = f"Based on the AI analysis, a {predicted_class_name} is detected. This is a malignant skin lesion that requires immediate medical attention."
                results["recommendations"] = ["Immediate consultation with a dermatologist", "Biopsy recommended for definitive diagnosis", "Avoid sun exposure and use high SPF sunscreen", "Regular skin examinations every 3-6 months"]
            else:
                results["description"] = f"Based on the AI analysis, a {predicted_class_name} is detected. This appears to be a benign lesion, but monitoring is recommended."
                results["recommendations"] = ["Continue regular skin self-exams", "Protect skin from sun exposure", "Monitor for any changes in size, color, or shape", "Consult dermatologist if any changes occur"]
                
        except Exception as e:
            return jsonify({"error": f"Skin cancer analysis failed: {e}"}), 500
    else:
        return jsonify({"error": "Invalid disease type or model not loaded"}), 400

    return jsonify(results)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API is running"""
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
    app.run(host="0.0.0.0", port=5000, debug=False)
    

