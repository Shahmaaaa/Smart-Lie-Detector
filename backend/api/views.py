# api/views.py
import os
import joblib   # ✅ use joblib instead of pickle
import subprocess
import numpy as np
import cv2
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array
import librosa
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
feature_extractor = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor.trainable = False

# --- 1. Load models once when the server starts ---
MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

try:
    # ✅ Load XGBoost vocal deception model (no space in filename)
    vocal_model = joblib.load(os.path.join(MODELS_DIR, "vocal_deception_model_xgboost.pkl"))

    # ✅ Load the scaler used during training
    vocal_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_xgboost.pkl"))

    # ✅ Load facial deep learning model
    facial_model = load_model(os.path.join(MODELS_DIR, "face.keras"))

except FileNotFoundError as e:
    print(f"🛑 CRITICAL ERROR: Could not load a model file. Make sure your model files are in 'models'. Error: {e}")
    vocal_model, vocal_scaler, facial_model = None, None, None


# --- 2. Helper functions ---
def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from a video file using ffmpeg."""
    try:
        command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -y -hide_banner -loglevel error "{audio_path}"'
        subprocess.run(command, shell=True, check=True)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def extract_vocal_features(audio_path, sr=22050, n_fft=512):
    """Extract robust features from audio for vocal deception detection."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=30)

        # If audio is too short, pad it
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), mode="constant")

        # --- Feature extraction ---
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft).T, axis=0)

        # Some features like tonnetz require harmonic component → guard with try
        try:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        except Exception:
            tonnetz = np.zeros(6)  # fallback if too short for tonnetz

        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        # --- Fix shape mismatch ---
        if features.ndim > 1:
            features = features.flatten()
        return features
    except Exception as e:
        print(f"Error extracting vocal features: {e}")
        return None


def process_video_for_facial_model(video_path):
    """Extract ONE frame from video and preprocess with VGG16 to match 25088 features."""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Resize and preprocess
        frame = cv2.resize(frame, (224, 224))
        frame = np.expand_dims(frame, axis=0)  # (1,224,224,3)
        frame = preprocess_input(frame)  # VGG16 preprocessing

        # Extract VGG16 features
        features = feature_extractor.predict(frame)
        features = features.flatten().reshape(1, -1)  # (1,25088)

        return features
    except Exception as e:
        print(f"Error processing video for facial model: {e}")
        return None




# --- 3. The API View ---
class AnalyzeVideoView(APIView):

    def post(self, request, *args, **kwargs): # type: ignore
        try:
            video_file = request.FILES.get('video')
            if not video_file:
                return Response({"error": "No video file provided"}, status=status.HTTP_400_BAD_REQUEST)

            temp_video_path = "temp_video_upload.mp4"
            with open(temp_video_path, 'wb+') as dest:
                for chunk in video_file.chunks():
                    dest.write(chunk)

            # --- Vocal Analysis ---
            vocal_truth_prob = 0.5
            temp_audio_path = extract_audio(temp_video_path)
            if not temp_audio_path:
                return Response({"error": "Failed to extract audio"}, status=status.HTTP_400_BAD_REQUEST)

            if vocal_model and vocal_scaler:
                vocal_features = extract_vocal_features(temp_audio_path)
                if vocal_features is None:
                    return Response({"error": "Failed to extract vocal features"}, status=status.HTTP_400_BAD_REQUEST)

                try:
                    scaled_features = vocal_scaler.transform(vocal_features.reshape(1, -1))
                    vocal_truth_prob = vocal_model.predict_proba(scaled_features)[0][1]
                except Exception as e:
                    return Response({"error": f"Vocal model prediction failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            # --- Facial Analysis ---
            facial_truth_prob = 0.5
            if facial_model:
                processed_video = process_video_for_facial_model(temp_video_path)
                if processed_video is None:
                    return Response({"error": "Failed to process video for facial model"}, status=status.HTTP_400_BAD_REQUEST)

                try:
                    facial_truth_prob = facial_model.predict(processed_video)[0][0]
                except Exception as e:
                    return Response({"error": f"Facial model prediction failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            # --- Combine Results ---
            final_truth_probability = (float(vocal_truth_prob) + float(facial_truth_prob)) / 2.0
            decision = "Truth Indicated" if final_truth_probability >= 0.5 else "Deception Indicated"
            confidence = final_truth_probability if decision == "Truth Indicated" else 1 - final_truth_probability

            return Response({
                "decision": decision,
                "confidence": f"{confidence * 100:.2f}%",
                "details": {
                    "vocalTruthProbability": f"{float(vocal_truth_prob) * 100:.2f}%",
                    "facialTruthProbability": f"{float(facial_truth_prob) * 100:.2f}%"
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # ✅ Global fallback for unexpected errors
            return Response({"error": f"Unexpected server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    def post(self, request, *args, **kwargs):
        video_file = request.FILES.get('video')
        if not video_file:
            return Response({"error": "No video file provided"}, status=status.HTTP_400_BAD_REQUEST)

        temp_video_path = "temp_video_upload.mp4"
        with open(temp_video_path, 'wb+') as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        # --- Vocal Analysis ---
        vocal_truth_prob = 0.5
        temp_audio_path = extract_audio(temp_video_path)
        if temp_audio_path and vocal_model and vocal_scaler:
            vocal_features = extract_vocal_features(temp_audio_path)
            if vocal_features is not None:
                scaled_features = vocal_scaler.transform(vocal_features.reshape(1, -1))
                vocal_truth_prob = vocal_model.predict_proba(scaled_features)[0][1]
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        # --- Facial Analysis ---
        facial_truth_prob = 0.5
        if facial_model:
            processed_video = process_video_for_facial_model(temp_video_path)
            if processed_video is not None:
                facial_truth_prob = facial_model.predict(processed_video)[0][0]

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        # --- Combine Results ---
        final_truth_probability = (float(vocal_truth_prob) + float(facial_truth_prob)) / 2.0
        decision = "Truth Indicated" if final_truth_probability >= 0.5 else "Deception Indicated"
        confidence = final_truth_probability if decision == "Truth Indicated" else 1 - final_truth_probability

        return Response({
            "decision": decision,
            "confidence": f"{confidence * 100:.2f}%",
            "details": {
                "vocalTruthProbability": f"{float(vocal_truth_prob) * 100:.2f}%",
                "facialTruthProbability": f"{float(facial_truth_prob) * 100:.2f}%"
            }
        }, status=status.HTTP_200_OK)
