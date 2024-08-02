import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Get the directory path of this script
script_dir = os.path.dirname(__file__)

# Path to the emotion model relative to the script's directory
model_filename = 'emotion_model.hdf5'
model_path = os.path.join(script_dir, model_filename)

# Load the pre-trained emotion recognition model
emotion_model = load_model(model_path)

# Emotion labels based on the pre-trained model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

def preprocess_input(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's built-in Haar cascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for emotion recognition
        face_processed = preprocess_input(face)
        
        # Predict the emotion
        emotion_prediction = emotion_model.predict(face_processed)
        max_index = np.argmax(emotion_prediction[0])
        emotion = emotion_labels[max_index]
        
        # Display the emotion label on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Display the frame with emotion labels
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
