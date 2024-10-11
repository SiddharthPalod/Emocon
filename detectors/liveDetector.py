import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def detect_emotions_from_video(model_path='../model/best_model.keras', scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    model = load_model(model_path)
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if process_this_frame:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #detects multiple faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
            
            face_emotions = []
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)

                preds = model.predict(roi_gray)[0]
                emotion_probability = np.max(preds)
                label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise'][np.argmax(preds)]
                face_emotions.append((x, y, w, h, label))

        process_this_frame = not process_this_frame

        # Display the results
        for (x, y, w, h, label) in face_emotions:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # Exit on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()
