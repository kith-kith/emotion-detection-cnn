# real_time_detection.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

#  Paths 
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "emotion_model1.keras")
train_dir = os.path.join(base_dir, "train")  # Needed for labels
emotion_labels = os.listdir(train_dir)

#Load Model
model = load_model(model_path)

#Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_size = 48

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        margin = 10
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = x + w + margin, y + h + margin
        roi = gray_frame[y1:y2, x1:x2]

        try:
            roi_resized = cv2.resize(roi, (img_size, img_size))
        except:
            continue

        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, img_size, img_size, 1))

        prediction = model.predict(roi_reshaped, verbose=0)
        label = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
