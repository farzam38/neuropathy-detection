from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dfu_model.h5')
model = load_model(MODEL_PATH)
class_names = ['Abnormal', 'Normal']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img = image.load_img(file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        preds = model.predict(x)
        class_idx = int(np.argmax(preds))
        class_name = class_names[class_idx]
        confidence = float(preds[0][class_idx])
        return jsonify({
            'prediction': class_name,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return 'Neuropathy DFU Model API is running.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 