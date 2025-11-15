import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from utils import speak

# Load the trained model and labels
model = tf.keras.models.load_model('model/traffic_sign_model.h5')
label_map = pd.read_csv('labels.csv')

# Prediction helper
def predict_sign(frame):
    try:
        img = cv2.resize(frame, (30, 30))
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)
        sign_name = label_map[label_map['ClassId'] == class_id]['SignName'].values[0]
        return sign_name, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0


# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Starting Real-Time Traffic Sign Detection...")
print("Press 'q' to quit.")

last_sign = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (optional)
    roi = cv2.resize(frame, (30, 30))
    sign_name, conf = predict_sign(roi)

    if conf > 0.8:  # threshold to avoid false predictions
        cv2.putText(frame, f"{sign_name} ({conf*100:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Voice feedback only when new sign detected
        if last_sign != sign_name:
            speak(f"Detected {sign_name}")
            last_sign = sign_name

    cv2.imshow("Traffic Sign Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
