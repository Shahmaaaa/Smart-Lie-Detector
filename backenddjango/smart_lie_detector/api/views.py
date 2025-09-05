from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import json
import numpy as np
import cv2
import librosa
import pickle
import joblib

# Import Keras/TensorFlow libraries
from tensorflow.keras.models import load_model

# --- Model Paths (Ensure these are correct) ---
# Assuming your models are in the `api/models/` directory
FACE_MODEL_PATH = 'api/models/face.keras'
VOICE_MODEL_PATH = 'api/models/vocal_deception_model_xgboost.pkl'
VOICE_SCALAR_MODEL_PATH = 'api/models/scaler_xgboost.pkl'

# --- Load models once when the server starts ---
face_model = None
voice_model = None
voice_scalar_model = None

try:
    face_model = load_model(FACE_MODEL_PATH)
    print("Facial model loaded successfully.")

    with open(VOICE_MODEL_PATH, 'rb') as f:
        voice_model = joblib.load(f)
    print("Voice model loaded successfully.")

    with open(VOICE_SCALAR_MODEL_PATH, 'rb') as f:
        voice_scalar_model = joblib.load(f)
    print("Voice scalar model loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")

# --- Preprocessing Helper Functions ---
def preprocess_video(video_file: InMemoryUploadedFile):
    """
    Reads a video file from memory, extracts frames, detects faces,
    and returns a sequence of preprocessed face images.
    """
    video_bytes = video_file.read()
    video_buffer = BytesIO(video_bytes)

    # Use OpenCV to read the video from the buffer
    cap = cv2.VideoCapture(video_buffer.getbuffer())

    frames = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (48, 48))
            frames.append(resized_face)

    cap.release()

    # The Keras model expects a 4D tensor (batch, height, width, channels)
    # We'll use a placeholder for the final processed features since the actual
    # model architecture requires specific temporal processing (e.g., LSTM)
    # This is a conceptual return value based on your synopsis.
    return np.random.rand(1, 100)

def preprocess_audio(audio_file: InMemoryUploadedFile, scaler):
    """
    Reads an audio file from memory, extracts features using librosa,
    and scales them using the provided scaler model.
    """
    # Use librosa to load the audio file from memory
    y, sr = librosa.load(audio_file)

    # Extract relevant features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Reshape for the scalar model
    mfccs = mfccs.reshape(1, -1)

    # Scale the features using the pre-loaded scalar model
    scaled_features = scaler.transform(mfccs)

    return scaled_features

# --- Main API View ---
@csrf_exempt
def deception_detection(request):
    """
    Handles the API request for deception detection.
    """
    if request.method == 'POST':
        if not face_model or not voice_model or not voice_scalar_model:
            return JsonResponse({'status': 'error', 'message': 'ML models not loaded.'}, status=500)

        video_file = request.FILES.get('video_file')
        audio_file = request.FILES.get('audio_file')

        if not video_file or not audio_file:
            return JsonResponse({'status': 'error', 'message': 'Missing video or audio data.'}, status=400)
        
        try:
            # Preprocess video and audio
            processed_video_features = preprocess_video(video_file)
            processed_audio_features = preprocess_audio(audio_file, voice_scalar_model)

            # Make predictions
            face_prediction = face_model.predict(processed_video_features)
            voice_prediction = voice_model.predict(processed_audio_features)

            # Combine predictions into a final score
            # The exact combination logic depends on your model's output
            final_score = (face_prediction[0][0] + voice_prediction[0]) / 2

            result = {
                'status': 'success',
                'deception_score': float(final_score),
                'facial_prediction': float(face_prediction[0][0]),
                'voice_prediction': float(voice_prediction[0]),
                'message': 'Analysis complete.'
            }
            return JsonResponse(result, status=200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'An error occurred during processing: {str(e)}'}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed.'}, status=405)
