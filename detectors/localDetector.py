import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import END

def detect_emotions_from_local_image(file_path, result_text, result_label):
    model = load_model('../model/best_model.keras')
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        result_text.insert(END, 'No faces detected\n')
        return
    else:
        result_text.insert(END, f"{len(faces)} face(s) detected, processing emotions...\n")

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
        result_text.insert(END, f"Face at ({x},{y}) - Emotion: {emotion}\n")        
    result_text.insert(END, f"Emotions detected: {', '.join(emotions)}\n")
    result_label.config(text="Image processing completed.")  

def detect_emotions_from_local_video(file_path, result_text, result_label):
    model = load_model('../model/best_model.keras')
    video_capture = cv2.VideoCapture(file_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if process_this_frame:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_emotions = []
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)

                preds = model.predict(roi_gray)[0]
                emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise'][np.argmax(preds)]
                face_emotions.append((x, y, w, h, emotion))

        process_this_frame = not process_this_frame
        for (x, y, w, h, label) in face_emotions:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    video_capture.release()
    cv2.destroyAllWindows()
    result_label.config(text="Video processing completed.")  # Update result text widget at the end
