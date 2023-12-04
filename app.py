from flask import Flask, request, jsonify
from keras.models import model_from_json
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)

# Load model from JSON and H5 files
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')

# Function to preprocess image and make predictions
def preprocess_image(image):
    # Resize image to (150, 150) and ensure 3 channels (RGB)
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    # Ensure image has 3 channels
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Normalize pixel values to be between 0 and 1
    image_array = image_array / 255.0

    return image_array

def get_glasses_frames(prediction):
    # Implement logic to convert prediction to glasses frames
    # ...

    # Placeholder logic: Using the index with the highest value
    selected_glasses_index = np.argmax(prediction)
    glasses_frames = [selected_glasses_index]

    return glasses_frames

# Map class index to label
class_labels = {
    0: "Diamond",
    1: "Heart",
    2: "Oblong",
    3: "Oval",
    4: "Round",
    5: "Square",
    6: "Triangle"
}

def predict_glasses(image):
    # Perform any necessary image preprocessing here
    processed_image = preprocess_image(image)

    # Perform prediction using the loaded model
    prediction = loaded_model.predict(np.array([processed_image]))

    # Convert prediction to glasses frames (replace with your logic)
    glasses_frames = get_glasses_frames(prediction)

    # Map the predicted class index to the corresponding label
    predicted_label = class_labels.get(glasses_frames[0], "Unknown")

    return f"Class: {predicted_label}, Label: {glasses_frames[0]}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from POST request
        image_file = request.files['image']
        image = Image.open(image_file)

        # Make prediction
        result = predict_glasses(image)

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
