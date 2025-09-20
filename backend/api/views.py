import os
import joblib 
import subprocess
import numpy as np
import cv2
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from keras.models import load_model 
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter

# Correct and Cleaned Imports for Keras Applications
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

# --- 1. GLOBAL FEATURE EXTRACTOR DEFINITION (Only VGG16 is needed) ---
VGG_INPUT_SHAPE = (224, 224, 3)
RESNET_INPUT_SHAPE = (224, 224, 3)
INCEPTION_INPUT_SHAPE = (224, 224, 3)  # FIXED: Match expected shape from error log

# VGG16 Feature Extractor (KEEP THIS ONE - used by facial_model_vgg/face.keras)
VGG_FEATURE_EXTRACTOR = VGG16(weights="imagenet", include_top=False, input_shape=VGG_INPUT_SHAPE)
VGG_FEATURE_EXTRACTOR.trainable = False 

# --- 2. Model Initialization ---
MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

# Global variables for all six models
vocal_model_xgb = None 
vocal_scaler_xgb = None
vocal_model_cnn = None
facial_model_vgg = None
facial_model_resnet = None
facial_model_inception = None

# Global variables for LAZY LOADING the large Wav2Vec2 model
WAV2VEC2_PROCESSOR = None
WAV2VEC2_MODEL = None
WAV2VEC2_PATH = os.path.join(MODELS_DIR, "wav2vec2_lie_detector_final")
WAV2VEC2_LOADED = False 

try:
    # --- Load Vocal Models Synchronously ---
    vocal_model_xgb = joblib.load(os.path.join(MODELS_DIR, "vocal_deception_model_xgboost.pkl"))
    vocal_scaler_xgb = joblib.load(os.path.join(MODELS_DIR, "scaler_xgboost.pkl"))
    vocal_model_cnn = load_model(os.path.join(MODELS_DIR, "deception_cnn_model_augmented.h5"))
    
    # --- Load Facial Models ---
    facial_model_vgg = load_model(os.path.join(MODELS_DIR, "face.keras"))
    facial_model_resnet = load_model(os.path.join(MODELS_DIR, "resnet50.keras"))
    facial_model_inception = load_model(os.path.join(MODELS_DIR, "inception.keras"))

except FileNotFoundError as e:
    print(f"🛑 CRITICAL ERROR: Could not load a model file. Error: {e}")


# ----------------------------------------------------------------------
# --- WAV2VEC2 LAZY LOADING ---
# ----------------------------------------------------------------------
def load_wav2vec2_once():
    """Loads Wav2Vec2 model and processor only if they haven't been loaded yet."""
    global WAV2VEC2_PROCESSOR, WAV2VEC2_MODEL, WAV2VEC2_LOADED
    
    if WAV2VEC2_LOADED:
        return True
    
    try:
        print("⏳ Loading Wav2Vec2 model for the first time. This may take a moment...")
        WAV2VEC2_PROCESSOR = Wav2Vec2Processor.from_pretrained(WAV2VEC2_PATH)
        WAV2VEC2_MODEL = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC2_PATH)
        WAV2VEC2_MODEL.eval()
        WAV2VEC2_LOADED = True
        print("✅ Wav2Vec2 model loaded successfully.")
        return True
    except Exception as e:
        print(f"🛑 Error loading Wav2Vec2 model: {e}")
        WAV2VEC2_LOADED = False
        return False

# ----------------------------------------------------------------------
# --- 3. Helper Functions (Vocal Functions) ---
# ----------------------------------------------------------------------

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from a video file using ffmpeg, forcing WAV format for compatibility."""
    try:
        # Force WAV output to avoid codec issues like Opus
        command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -f wav -y -hide_banner -loglevel error "{audio_path}"'
        result = subprocess.run(command, shell=True, check=True, capture_output=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr.decode()}")
            return None
        print(f"Audio extracted successfully to {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_vocal_features_xgb(audio_path, sr=16000, n_fft=512):
    """Extract robust features (MFCC, Chroma, etc.) for XGBoost model."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=30) 
        
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), mode="constant")

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft).T, axis=0)
        
        try:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        except Exception:
            tonnetz = np.zeros(6) 

        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        if features.ndim > 1:
            features = features.flatten()
        return features
    except Exception as e:
        print(f"Error extracting XGBoost features: {e}")
        return None

def extract_cnn_spectrogram_features(audio_path, sr=16000, n_mels=128, max_len=500):
    """Extracts a Mel Spectrogram image (128x500x1) for the CNN model."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=5.0) 
        
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        
        if melspec_db.shape[1] > max_len:
            melspec_db = melspec_db[:, :max_len]
        elif melspec_db.shape[1] < max_len:
            pad_width = max_len - melspec_db.shape[1]
            melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

        melspec_db = (melspec_db - np.min(melspec_db)) / (np.max(melspec_db) - np.min(melspec_db) + 1e-6)
        
        features = melspec_db[np.newaxis, :, :, np.newaxis]
        
        return features
    except Exception as e:
        print(f"Error extracting CNN Spectrogram features: {e}")
        return None

def predict_wav2vec2(audio_path):
    """Predict deception probability using the Wav2Vec2 model."""
    if not load_wav2vec2_once():
        return {'prob': 0.5, 'class': 1} 
    
    try:
        y, sr = librosa.load(audio_path, sr=16000) 
        print(f"Audio length: {len(y)} samples")  # Debug audio length
        if len(y) == 0:
            print("Empty audio - skipping Wav2Vec2")
            return {'prob': 0.5, 'class': 1}
        
        MAX_SAMPLES = 160000  # Increased
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]  # Truncate
        
        if WAV2VEC2_PROCESSOR is None or WAV2VEC2_MODEL is None:
            raise RuntimeError("Wav2Vec2 processor or model failed to load.")

        inputs = WAV2VEC2_PROCESSOR(
            [y], 
            sampling_rate=16000, # type: ignore
            return_tensors="pt", # type: ignore
            padding=True, # type: ignore
            truncation=True # type: ignore
        )

        with torch.no_grad():
            logits = WAV2VEC2_MODEL(**inputs).logits
        
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        
        return {
            'prob': probabilities[1], 
            'class': np.argmax(logits.numpy(), axis=1)[0]
        }
        
    except Exception as e:
        print(f"Error predicting with Wav2Vec2 (check audio format/codec): {e}")
        return {'prob': 0.5, 'class': 1} 

# ----------------------------------------------------------------------
# --- 4. FACIAL HELPER FUNCTIONS ---
# ----------------------------------------------------------------------

def extract_frame(video_path):
    """Extract ONE raw frame (no resize yet)."""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None

def get_facial_prediction(model, frame, preprocess_func, model_name, input_shape):
    """Utility function to resize, preprocess, predict for one facial model."""
    
    global VGG_FEATURE_EXTRACTOR 
    
    if model is None or frame is None:
        return {'prob': 0.5, 'class': 1}

    try:
        # Resize to model-specific shape
        resized_frame = cv2.resize(frame, input_shape[:2])
        
        # Add batch dimension and run Keras preprocessing
        preprocessed_image = preprocess_func(np.expand_dims(resized_frame, axis=0))

        # Normalization
        
        # --- Conditional Input Data Flow ---
        if model_name == 'VGG16':
            # VGG16 uses features
            features = VGG_FEATURE_EXTRACTOR.predict(preprocessed_image, verbose=0)
            input_data = features.flatten().reshape(1, -1)
            
        elif model_name in ['ResNet50', 'Inception']:
            # ResNet50/Inception expect the preprocessed/normalized IMAGE
            input_data = preprocessed_image
            
        else:
            raise ValueError(f"Unknown facial model name: {model_name}")
        
        # Predict using the model
        prediction = model.predict(input_data, verbose=0)
        
        prob = prediction[0][0] 
        cls = 1 if prob >= 0.5 else 0
        
        return {'prob': float(prob), 'class': cls}
        
    except Exception as e:
        print(f"Error predicting with {model_name}: {e}")
        return {'prob': 0.5, 'class': 1}

# ----------------------------------------------------------------------
# --- 5. The API View ---
# ----------------------------------------------------------------------
class AnalyzeVideoView(APIView):

    def post(self, request, *args, **kwargs):
        
        results = {
            'xgb': {'prob': 0.5, 'class': 1}, 
            'w2v2': {'prob': 0.5, 'class': 1},
            'cnn': {'prob': 0.5, 'class': 1}, 
            'vgg': {'prob': 0.5, 'class': 1},
            'resnet': {'prob': 0.5, 'class': 1},
        }
        
        temp_video_path = "temp_video_upload.mp4"
        temp_audio_path = "temp_audio.wav"

        try:
            video_file = request.FILES.get('video')
            if not video_file:
                return Response({"error": "No video file provided"}, status=status.HTTP_400_BAD_REQUEST)

            with open(temp_video_path, 'wb+') as dest:
                for chunk in video_file.chunks():
                    dest.write(chunk)

            # --- VOCAL ANALYSIS ---
            temp_audio_path = extract_audio(temp_video_path)
            if not temp_audio_path:
                return Response({"error": "Failed to extract audio"}, status=status.HTTP_400_BAD_REQUEST)
            
            vocal_features_xgb = extract_vocal_features_xgb(temp_audio_path, sr=16000) if vocal_model_xgb else None
            cnn_features = extract_cnn_spectrogram_features(temp_audio_path, sr=16000) if vocal_model_cnn else None
            
            # XGBoost Prediction
            if vocal_model_xgb and vocal_scaler_xgb and vocal_features_xgb is not None:
                scaled_features = vocal_scaler_xgb.transform(vocal_features_xgb.reshape(1, -1))
                results['xgb']['prob'] = vocal_model_xgb.predict_proba(scaled_features)[0][1]
                results['xgb']['class'] = vocal_model_xgb.predict(scaled_features)[0]

            # Wav2Vec2 Prediction
            if WAV2VEC2_MODEL and WAV2VEC2_PROCESSOR:
                results['w2v2'] = predict_wav2vec2(temp_audio_path)

            # Vocal CNN Prediction
            if vocal_model_cnn and cnn_features is not None:
                try:
                    cnn_prob = vocal_model_cnn.predict(cnn_features, verbose=0)[0][0] 
                    results['cnn']['prob'] = float(cnn_prob)
                    results['cnn']['class'] = 1 if cnn_prob >= 0.5 else 0
                except Exception as e:
                    print(f"Vocal CNN model prediction failed: {e}")
            
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            # --- VOCAL ENSEMBLE (Soft Voting) ---
            vocal_probs = [results['xgb']['prob'], results['w2v2']['prob'], results['cnn']['prob']]
            final_vocal_prob = np.mean(vocal_probs)
            final_vocal_class = 1 if final_vocal_prob >= 0.5 else 0

            # --- FACIAL ANALYSIS (Weighted Average of VGG16 and ResNet50) ---
            raw_frame = extract_frame(temp_video_path)
            results['vgg'] = get_facial_prediction(facial_model_vgg, raw_frame, vgg16_preprocess, 'VGG16', VGG_INPUT_SHAPE)
            results['resnet'] = get_facial_prediction(facial_model_resnet, raw_frame, resnet50_preprocess, 'ResNet50', RESNET_INPUT_SHAPE)

            final_facial_prob = 0.5 * results['vgg']['prob'] + 0.5 * results['resnet']['prob']
            final_facial_class = 1 if final_facial_prob >= 0.5 else 0

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            # --- FINAL ENSEMBLE (Weighted Average) ---
           # --- ✅ FINAL ENSEMBLE DECISION (Improved Logic) ---

            if final_vocal_class == final_facial_class:
            # 🤝 AGREEMENT MODE
                final_class = final_vocal_class
                final_truth_probability = (final_vocal_prob + final_facial_prob) / 2
                decision = "Truth Indicated" if final_class == 1 else "Deception Indicated"
                confidence = final_truth_probability if final_class == 1 else 1 - final_truth_probability
                print("🤝 AGREEMENT MODE: Both Vocal & Facial models agree.")
            else:
            # ⚖️ DISAGREEMENT MODE - fallback to average probability
                final_truth_probability = 0.5 * final_vocal_prob + 0.5 * final_facial_prob
                final_class = 1 if final_truth_probability >= 0.5 else 0
                decision = "Truth Indicated" if final_class == 1 else "Deception Indicated"
                confidence = final_truth_probability if final_class == 1 else 1 - final_truth_probability
                print("⚖️ DISAGREEMENT MODE: Models disagree, using average probability fallback.")

            # --- PRINT EVERYTHING TO TERMINAL ---
            print("\n" + "="*60)
            print("🚀 VOCAL MODEL RESULTS 🚀")
            print(f"XGBoost (MFCC) -> Class: {results['xgb']['class']} | Prob: {results['xgb']['prob']:.4f}")
            print(f"Wav2Vec2      -> Class: {results['w2v2']['class']} | Prob: {results['w2v2']['prob']:.4f}")
            print(f"Vocal CNN H5  -> Class: {results['cnn']['class']} | Prob: {results['cnn']['prob']:.4f}")
            print(f"VOCAL ENSEMBLE -> Class: {final_vocal_class} | Prob: {final_vocal_prob:.4f}")
            print("-"*60)
            print("👁️ FACIAL MODEL RESULTS 👁️")
            print(f"VGG16   -> Prob: {results['vgg']['prob']:.4f}")
            print(f"ResNet50-> Prob: {results['resnet']['prob']:.4f}")
            print(f"FACIAL AVERAGE -> Class: {final_facial_class} | Prob: {final_facial_prob:.4f}")
            print("-"*60)
            print("🏁 FINAL ENSEMBLE DECISION 🏁")
            print(f"Decision: {decision} | Confidence: {confidence:.4f}")
            print("="*60 + "\n")

            return Response({
                "decision": decision,
                "confidence": f"{confidence * 100:.2f}%",
                "details": {
                    "vocalEnsembleDecision": "Truth Indicated" if final_vocal_class == 1 else "Deception Indicated",
                    "vocalEnsembleTruthProbability": f"{final_vocal_prob * 100:.2f}%",
                    "facialAverageDecision": "Truth Indicated" if final_facial_class == 1 else "Deception Indicated",
                    "facialAverageTruthProbability": f"{final_facial_prob * 100:.2f}%",
                    "vocalModelBreakdown": {
                        "XGBoost_MFCC": f"Class {results['xgb']['class']} | Prob {results['xgb']['prob'] * 100:.2f}%",
                        "Wav2Vec2": f"Class {results['w2v2']['class']} | Prob {results['w2v2']['prob'] * 100:.2f}%",
                        "Vocal_CNN_H5": f"Class {results['cnn']['class']} | Prob {results['cnn']['prob'] * 100:.2f}%",
                    },
                    "facialModelBreakdown": {
                        "VGG16": f"Prob {results['vgg']['prob'] * 100:.2f}%",
                        "ResNet50": f"Prob {results['resnet']['prob'] * 100:.2f}%",
                    }
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return Response({"error": f"Unexpected server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

