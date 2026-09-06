import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = (64, 64)

def predict_flower(image_path, model_path='model/flower_rnn.keras'):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Run train.py first.")
        return

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess Image
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Reshape into RNN sequence: (64, 64, 3) -> (64, 192)
    rnn_sequence = img_array.reshape(64, 192)
    input_data = np.expand_dims(rnn_sequence, axis=0)  # Shape: (1, 64, 192)

    # Predict
    predictions = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100

    print(f"\nResult:")
    print(f"Predicted Flower : {CLASS_NAMES[predicted_idx].capitalize()}")
    print(f"Confidence       : {confidence:.2f}%\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_flower(sys.argv[1])