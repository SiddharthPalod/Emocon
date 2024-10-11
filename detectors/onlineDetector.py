import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk

def detect_emotions_from_online_image(url, result_text):
    try:
        response = requests.get(url)
        img_data = response.content
        
        img = Image.open(BytesIO(img_data))
        img = np.array(img)  # Convert the image to a NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        model = load_model('../model/best_model.keras')
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            result_text.insert(tk.END, 'No faces detected\n')
            return
        else:
            result_text.insert(tk.END, f"{len(faces)} face(s) detected, processing emotions...\n")
        emotions = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = face / 255.0
            preds = model.predict(face)
            emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise'][np.argmax(preds)]
            emotions.append(emotion)
            result_text.insert(tk.END, f"Face at ({x},{y}) - Emotion: {emotion}\n")
        result_text.insert(tk.END, f"Emotions detected: {', '.join(emotions)}\n")
    except Exception as e:
        result_text.insert(tk.END, f"Error processing URL: {e}\n")
