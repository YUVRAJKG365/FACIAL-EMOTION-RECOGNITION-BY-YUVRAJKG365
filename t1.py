import cv2
import os
import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Suppress deprecation and other warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Corrected paths
model_dir = r'C:\Users\yuvra\Documents\pp'  # Correct model directory
model_name = 'yuvrajkg_model.keras'  # Model file name
model_path = os.path.join(model_dir, model_name)  # Complete model path

data_dir = r'C:\Users\yuvra\Documents\pp\collected_data'  # Correct data directory

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(r'C:/Users/yuvra/Documents/pp/haarface.xml')  # Correct path

# Ensure data directory exists for saving new data
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load pre-trained model
try:
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure the model exists or train it first.")
        exit()
    model = load_model(model_path)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {e}")
    exit()

# Define emotion classes and corresponding emojis
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😞',
    'surprise': '😮'
}

# Define emotion scores (0-5)
emotion_scores = {
    'angry': 1,
    'disgust': 1,
    'fear': 2,
    'happy': 5,
    'neutral': 3,
    'sad': 1,
    'surprise': 4
}

def preprocess_face(face):
    """
    Preprocess the face region for emotion classification.
    """
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (48, 48))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def save_face_data(face, emotion_label):
    """
    Save collected face data for future retraining.
    """
    label_dir = os.path.join(data_dir, emotion_label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    count = len(os.listdir(label_dir))
    file_path = os.path.join(label_dir, f"{count + 1}.png")
    cv2.imwrite(file_path, face)

def retrain_model():
    """
    Retrain the model with new data.
    """
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_data = data_gen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = data_gen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Retrain the existing model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    # Save the updated model
    model.save(model_path)

def camera_loop():
    """
    Camera loop for real-time emotion detection.
    """
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    # Create containers for dynamic updates
    emotion_label_container = st.empty()
    emoji_container = st.empty()
    score_container = st.empty()

    while st.session_state['camera_on']:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Please check your camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            face_preprocessed = preprocess_face(face)
            prediction = model.predict(face_preprocessed)
            class_idx = np.argmax(prediction)

            emotion = emotion_classes[class_idx]
            emoji = emotion_emojis.get(emotion, '😐')
            score = emotion_scores.get(emotion, 0)

            emotion_label_container.markdown(f"**Detected Emotion:** {emotion}")
            emoji_container.markdown(f"**Emoji:** {emoji}")
            score_container.markdown(f"**Emotion Score:** {score}/5")

            save_face_data(face, emotion)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    stframe.empty()

def main():
    st.title('''Facial Emotion Recognition
    by Yuvraj Gond''')

    if 'camera_on' not in st.session_state:
        st.session_state['camera_on'] = False

    if st.button("Start Camera" if not st.session_state['camera_on'] else "Stop Camera"):
        st.session_state['camera_on'] = not st.session_state['camera_on']
        if st.session_state['camera_on']:
            camera_loop()

if __name__ == '__main__':
    main()



# Proprietary License - All Rights Reserved
# Copyright (C) 2025 Yuvraj Kumar Gond
# Unauthorized copying or distribution of this file is strictly prohibited.

