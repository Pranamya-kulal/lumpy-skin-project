from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

# Load your trained model
model = tf.keras.models.load_model('my_modelll.keras')

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Lumpy Skin Disease Detection API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     try:
#         # Open the image file
#         img = Image.open(file.stream)
        
#         # Preprocess the image (resize and normalize)
#         img = img.resize((128, 128))  # Resize the image to 128x128
#         img_array = np.array(img)
        
#         # Check if the image is grayscale (it may have 1 channel)
#         if len(img_array.shape) == 2:  # If it's grayscale
#             img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB by repeating the grayscale channels
        
#         # Normalize the image
#         img_array = img_array / 255.0  # Normalize if needed
        
#         # Add batch dimension (reshape to (1, 128, 128, 3))
#         img_array = np.expand_dims(img_array, axis=0)
        
#         # Predict using the model
#         predictions = model.predict(img_array)
        
#         predicted_class = predictions[0][0]  # Assuming binary classification
        
#         # If predicted value > 0.5, classify as "Lumpy Skin Disease"
#         if predicted_class > 0.5:
#             prediction_label = "Lumpy Skin Disease"
#         else:
#             prediction_label = "No Lumpy Skin Disease"
        
#         # Return the prediction result
#         return jsonify({'prediction': prediction_label, 'confidence': str(predicted_class)})
    
#     except Exception as e:
#         return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Save the uploaded file to a folder (e.g., 'uploads/')
        filename = secure_filename(file.filename)
        upload_folder = 'uploads/'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Open the image file
        img = Image.open(file_path)
        
        # Preprocess the image (resize and normalize)
        img = img.resize((128, 128))  # Resize the image to 128x128
        img_array = np.array(img)
        
        # Check if the image is grayscale (it may have 1 channel)
        if len(img_array.shape) == 2:  # If it's grayscale
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB by repeating the grayscale channels
        
        # Normalize the image
        img_array = img_array / 255.0  # Normalize if needed
        
        # Add batch dimension (reshape to (1, 128, 128, 3))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict using the model
        predictions = model.predict(img_array)
        
        predicted_class = predictions[0][0]  # Assuming binary classification
        
        # If predicted value > 0.5, classify as "Lumpy Skin Disease"
        if predicted_class > 0.5:
            prediction_label = "Lumpy Skin Disease"
        else:
            prediction_label = "No Lumpy Skin Disease"
        
        # Return the prediction result along with the image URL
        image_url = f'http://127.0.0.1:5000/uploads/{filename}'
        return jsonify({
            'prediction': prediction_label,
            'confidence': str(predicted_class),
            'image_url': image_url
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import os
# from werkzeug.utils import secure_filename

# Load your trained model
# model = tf.keras.models.load_model('lumpy_skin_disease_model.keras')

# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def home():
#     return "Lumpy Skin Disease Detection API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     try:
#         Save the uploaded file to a folder (e.g., 'uploads/')
#         filename = secure_filename(file.filename)
#         upload_folder = 'uploads/'
#         if not os.path.exists(upload_folder):
#             os.makedirs(upload_folder)
        
#         file_path = os.path.join(upload_folder, filename)
#         file.save(file_path)
        
#         Open the image file
#         img = Image.open(file_path)
        
#         Preprocess the image (resize and normalize)
#         img = img.resize((224, 224))  # Update to match model input size
#         img_array = np.array(img)
        
#         Check if the image is grayscale (it may have 1 channel)
#         if len(img_array.shape) == 2:  # If it's grayscale
#             img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB by repeating the grayscale channels
        
#         Normalize the image
#         img_array = img_array / 255.0  # Normalize if needed
        
#         Add batch dimension (reshape to (1, 224, 224, 3))
#         img_array = np.expand_dims(img_array, axis=0)
        
#         Predict using the model
#         predictions = model.predict(img_array)
        
#         predicted_class = predictions[0][0]  # Assuming binary classification
        
#         If predicted value > 0.5, classify as "Lumpy Skin Disease"
#         if predicted_class > 0.5:
#             prediction_label = "Lumpy Skin Disease"
#         else:
#             prediction_label = "No Lumpy Skin Disease"
        
#         If the prediction is "No Lumpy Skin Disease" and confidence is low, assume non-cattle image
#         if predicted_class < 0.1:
#             prediction_label = "This is not a cattle image"
        
#         Return the prediction result along with the image URL
#         image_url = f'http://127.0.0.1:5000/uploads/{filename}'
#         return jsonify({
#             'prediction': prediction_label,
#             'confidence': str(predicted_class),
#             'image_url': image_url
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
