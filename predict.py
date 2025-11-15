import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from utils import speak

# Load model and labels
model = tf.keras.models.load_model('model/traffic_sign_model.h5')
label_map = pd.read_csv('labels.csv')

def predict_sign(image):
    img = cv2.resize(image, (30, 30))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    sign_name = label_map[label_map['ClassId'] == class_id]['SignName'].values[0]
    speak(f"Detected: {sign_name}")
    return sign_name
