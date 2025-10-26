# Emotion Detection using CNN

[![Python](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-green)](https://opencv.org/)

---

## Description

A real-time emotion detection system using a Convolutional Neural Network (CNN).
It detects faces from a webcam feed and classifies emotions using a trained CNN model.

**Key Points:**

* Real-time emotion recognition
* Uses grayscale images for faster processing
* CNN trained with custom datasets
* Saves models in `.keras` and `.h5` formats

---

## Features

* Live webcam face detection with OpenCV
* Emotion classification with CNN
* Data augmentation for robust training
* Train on your own dataset

---

## Requirements

* Python 3.x
* TensorFlow
* OpenCV
* Numpy

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
numpy
opencv-python
tensorflow
```

---

## Project Structure

```
emotion-detection-cnn/
│
├─ train/                # Training dataset (organized by emotion labels)
├─ validation/           # Validation dataset
├─ test/                 # Test dataset
├─ emotion_model1.keras  # Trained CNN model
├─ emotion_model1.h5     # Optional H5 model
├─ train_model.py        # Script to train CNN model
├─ real_time_detection.py# Script for real-time emotion detection
├─ README.md             # Project documentation
└─ requirements.txt      # Python dependencies
```

---

## Usage

### 1️⃣ Train the Model

```bash
python train_model.py
```

* Trains a CNN on your dataset.
* Saves the best model as `emotion_model1.keras`.
* Optionally saves a `.h5` version.

### 2️⃣ Real-Time Emotion Detection

```bash
python real_time_detection.py
```

* Opens webcam.
* Detects faces and predicts emotion in real-time.
* Press `q` to quit.

---

## Model Architecture

* **Conv2D + BatchNormalization + MaxPooling2D** layers
* **GlobalAveragePooling2D**
* **Dense layers** with Dropout
* **Softmax output layer** (number of classes = number of emotions in your dataset)

---

## Dataset Structure

Your datasets should be organized as:

```
train/
  happy/
  sad/
  angry/
  ...
validation/
  happy/
  sad/
  angry/
  ...
test/
  happy/
  sad/
  angry/
  ...
```

---

## Notes

* Ensure your webcam is connected for real-time detection.
* Grayscale images are used for efficiency.
* Adjust `img_size` or `batch_size` in scripts if needed.

---
## Demo

![Emotion Detection Demo](assets/demo.gif)

## License

This project is licensed under the MIT License.
