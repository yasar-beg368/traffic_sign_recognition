# 🚦 Traffic Sign Recognition System (with Real-Time Detection + Voice Alerts)

This project is a **Computer Vision-based Traffic Sign Recognition System** that detects and classifies real-world traffic signs using a **Convolutional Neural Network (CNN)**. The system also includes **real-time webcam detection** and **voice alerts** for driver assistance.

---

## 📌 Project Overview

Traffic sign recognition is a crucial component of advanced driver-assistance systems (ADAS) and autonomous driving. This project uses:

* **GTSRB Dataset (German Traffic Sign Recognition Benchmark)**
* A custom **CNN model** built with **TensorFlow/Keras**
* **OpenCV** for image processing
* **Streamlit** for web-based demo
* **pyttsx3** for real-time voice alerts

The system supports both:

* **Offline prediction** (upload an image)
* **Real-time webcam detection** (continuous recognition + audio feedback)

---

## 📂 Project Structure

```
traffic_sign_recognition/
│
├── data/
│   ├── Train/
│   ├── Test/
│   ├── Meta/
│   └── archive.zip
│
├── model/
│   └── traffic_sign_model.h5
│
├── app.py
├── train_model.py
├── realtime_detect.py
├── predict.py
├── unzip_data.py
├── labels.csv
└── requirements.txt
```

---

## 🧠 Model Architecture

A simple but powerful CNN architecture was used:

* **Conv2D → ReLU → MaxPool2D**
* **Conv2D → ReLU → MaxPool2D**
* **Flatten**
* **Dense → Dropout**
* **Softmax output layer (43 classes)**

The model achieved **~96% accuracy** on the validation set.

---

## 🗂 Dataset Used

**GTSRB: German Traffic Sign Recognition Benchmark**

* 43 classes
* 39,000+ training images
* Different lighting, angles, distortions

After extraction, the dataset folders should look like:

```
data/
├── Train/
├── Test/
├── Meta/
└── Train.csv / Test.csv / Meta.csv
```

---

## 🚀 Features

### ✔️ Train a CNN model on the GTSRB dataset

### ✔️ Real-time webcam detection using OpenCV

### ✔️ Voice alerts based on recognized signs

### ✔️ Streamlit web interface (image upload → detection)

### ✔️ Modular code structure

### ✔️ High accuracy (~96%)

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```
git clone <your_repo_url>
cd traffic_sign_recognition
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Unzip the dataset

Run:

```
python unzip_data.py
```

---

## 🧪 Training the Model

```
python train_model.py
```

This will:

* Load all images from `data/Train/`
* Train the CNN
* Save the model to `model/traffic_sign_model.h5`

---

## 🌐 Running the Streamlit Web App

```
streamlit run app.py
```

Upload any traffic sign image → Get prediction + voice alert.

---

## 🎥 Running Real-Time Webcam Detection

```
python realtime_detect.py
```

Features:

* Live detection
* Confidence score
* Voice alert only when a new sign is detected
* Press **q** to quit

---

## 📊 Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~96%  |
| Classes  | 43    |
| Dataset  | GTSRB |

---

## 📌 Technologies Used

* **TensorFlow** / **Keras**
* **OpenCV**
* **Streamlit**
* **pyttsx3** (Text-to-Speech)
* **NumPy**, **Pandas**, **Matplotlib**

---

## 📈 Future Improvements

* Integrate YOLOv8 for detection + classification
* Add object tracking (Deep SORT)
* Deploy mobile app (TFLite model)
* Improve FPS using multi-threading

---

## 👨‍💻 Author

**YASAR BEG**
Traffic Sign Recognition System — Computer Vision + Deep Learning + Real-Time Processing

---


