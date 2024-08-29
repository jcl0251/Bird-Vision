import tensorflow as tf
from tensorflow.keras import layers, datasets, models #type: ignore
import numpy as np
from nabirds import images_dir, class_id_to_name
from nabirds.data_loader import load_data
from nabirds.model import create_model
import cv2

model = tf.keras.model.load_model('final_model.keras')

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)  # Preprocess image
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_id_to_name[predicted_class]

# Example of predicting a new image
predicted_label = predict_image('path_to_new_image.jpg')
print(f"The model predicts: {predicted_label}")