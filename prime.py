import streamlit as st
import joblib
import numpy as np
import librosa

# Load the trained model
model = joblib.load(r"C:\Users\user\OneDrive\Desktop\voice reg\modelfile.pkl")

# Function to extract features from an audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = np.hstack([
        np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=43), axis=1),
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1),
        np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)[:10],
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.rms(y=y)),
        np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3), axis=1)
    ])
    return features.reshape(1, -1)

# Streamlit UI
st.title("Gender Recognition by Voice")

# Upload Audio File
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features
    features = extract_features("temp.wav")

    # Predict Gender
    prediction = model.predict(features)
    st.write(f"Predicted Gender: {prediction[0]}")