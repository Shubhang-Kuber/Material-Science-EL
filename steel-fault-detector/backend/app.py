"""
Steel Plate Fault Detection API
Flask backend with CNN model for image-based fault classification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Fault types from NEU-DET dataset (6 classes)
FAULT_TYPES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]

# Fault descriptions for user-friendly output
FAULT_DESCRIPTIONS = {
    'crazing': 'Fine network of surface cracks caused by thermal stress or material fatigue during cooling.',
    'inclusion': 'Foreign particles or non-metallic materials trapped within the steel during manufacturing.',
    'patches': 'Irregular surface areas with different texture or appearance from surrounding material.',
    'pitted_surface': 'Small holes or depressions on the steel surface caused by corrosion or defects.',
    'rolled-in_scale': 'Oxide scale that gets pressed into the surface during the rolling process.',
    'scratches': 'Linear marks or grooves on the surface caused by mechanical contact or handling.'
}

# Global model variable
model = None

def load_cnn_model():
    """Load the trained CNN model"""
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'steel_fault_cnn.h5')
    
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model not found. Please train the model first.")
        model = None

def preprocess_image(img_bytes):
    """Preprocess the uploaded image for CNN prediction"""
    # Open image from bytes
    img = Image.open(io.BytesIO(img_bytes))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    return img_array

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'fault_types': FAULT_TYPES
    })

@app.route('/api/fault-types', methods=['GET'])
def get_fault_types():
    """Get all fault types with descriptions"""
    fault_info = []
    for fault in FAULT_TYPES:
        fault_info.append({
            'name': fault,
            'description': FAULT_DESCRIPTIONS[fault]
        })
    return jsonify({'fault_types': fault_info})

@app.route('/api/predict', methods=['POST'])
def predict_fault():
    """Predict fault type from uploaded steel plate image"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500
    
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'success': False
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'error': 'No image selected',
            'success': False
        }), 400
    
    try:
        # Read and preprocess the image
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)
        
        # Make prediction
        predictions = model.predict(processed_img)[0]
        
        # Get top predictions
        results = []
        for i, (fault, confidence) in enumerate(zip(FAULT_TYPES, predictions)):
            results.append({
                'fault_type': fault,
                'confidence': float(confidence * 100),
                'description': FAULT_DESCRIPTIONS[fault]
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get the primary prediction
        primary_prediction = results[0]
        
        return jsonify({
            'success': True,
            'primary_fault': primary_prediction['fault_type'],
            'primary_confidence': primary_prediction['confidence'],
            'primary_description': primary_prediction['description'],
            'all_predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }), 500

@app.route('/api/demo-predict', methods=['POST'])
def demo_predict():
    """Demo prediction endpoint (works without trained model)"""
    
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'success': False
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'error': 'No image selected',
            'success': False
        }), 400
    
    try:
        # Generate demo predictions (random for demonstration)
        np.random.seed(42)
        predictions = np.random.dirichlet(np.ones(7)) * 100
        
        results = []
        for fault, confidence in zip(FAULT_TYPES, predictions):
            results.append({
                'fault_type': fault,
                'confidence': float(confidence),
                'description': FAULT_DESCRIPTIONS[fault]
            })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        primary_prediction = results[0]
        
        return jsonify({
            'success': True,
            'demo_mode': True,
            'primary_fault': primary_prediction['fault_type'],
            'primary_confidence': primary_prediction['confidence'],
            'primary_description': primary_prediction['description'],
            'all_predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Steel Plate Fault Detection API...")
    print("📊 Available fault types:", FAULT_TYPES)
    
    # Try to load the model
    load_cnn_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
