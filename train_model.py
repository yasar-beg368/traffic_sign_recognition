import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os

# Path to dataset
data_dir = "data/Train"

images = []
labels = []

for i in range(43):  # 43 classes in GTSRB
    path = os.path.join(data_dir, str(i))
    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.resize(img, (30, 30))
            images.append(img)
            labels.append(i)
        except:
            continue

images = np.array(images)
labels = np.array(labels)

# Normalize
X = images / 255.0
y = tf.keras.utils.to_categorical(labels, 43)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(30,30,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(X_train)

# Train
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=100,
                    validation_data=(X_val, y_val))

model.save('model/traffic_sign_model.h5')
print("✅ Model trained and saved successfully.")
