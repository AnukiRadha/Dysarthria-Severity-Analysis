"""
predict.py - Inference Engine for Dysarthria Speech Prediction
--------------------------------------------------------------
Processes uploaded or recorded audio files, extracts speech representations
using the selected speech transformer, and predicts the dysarthria severity.
"""

import os
import glob
import numpy as np
import audio
import model


def get_latest_trained_model(preferred_model_name=None):
    """
    Finds the most suitable trained model package in saved_models/.
    """
    saved_files = glob.glob("saved_models/*.joblib")
    if not saved_files:
        return None

    if preferred_model_name:
        clean_tag = preferred_model_name.lower().replace(" ", "_")
        for f in saved_files:
            if clean_tag in f.lower():
                return f

    # Fallback to the latest modified file
    saved_files.sort(key=os.path.getmtime, reverse=True)
    return saved_files[0]


def get_all_trained_models():
    """
    Lists all available trained model packages in saved_models/.
    """
    saved_files = glob.glob("saved_models/*.joblib")
    return sorted(saved_files, key=os.path.getmtime, reverse=True)


def predict_speech(audio_source, model_path=None, preferred_transformer="Wav2Vec 2.0"):
    """
    Performs end-to-end inference on a single audio recording.
    
    Parameters:
    - audio_source: file path, file-like object, or bytes
    - model_path: specific path to a saved .joblib file, or None to auto-detect
    - preferred_transformer: fallback transformer if not specified in model package
    
    Returns:
    - result: dict containing predicted class, confidence, probabilities, and audio metrics.
    """
    if model_path is None or not os.path.exists(model_path):
        model_path = get_latest_trained_model(preferred_transformer)
        if model_path is None:
            return {
                "success": False,
                "error": "No trained model found! Please train a model first on Page 3 (Model Training)."
            }

    try:
        # Load the saved model bundle
        package = model.load_trained_model(model_path)
        pipeline = package["pipeline"]
        label_encoder = package["label_encoder"]
        active_transformer = package.get("model_name", preferred_transformer)
        classes = package.get("classes", list(label_encoder.classes_))

        # 1. Load and preprocess audio to 16 kHz mono
        waveform, sr = audio.load_audio(audio_source)
        duration_sec = len(waveform) / sr

        # 2. Extract transformer embedding
        embedding = model.extract_embedding_from_waveform(waveform, model_name=active_transformer)
        embedding_2d = embedding.reshape(1, -1)

        # 3. Model prediction
        pred_idx = pipeline.predict(embedding_2d)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        # 4. Probabilities / confidence scores
        confidence = None
        probabilities = {}
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(embedding_2d)[0]
            confidence = float(np.max(probs))
            for cls_name, p in zip(classes, probs):
                probabilities[cls_name] = round(float(p) * 100, 2)
        else:
            confidence = 1.0
            probabilities = {cls_name: 100.0 if cls_name == pred_label else 0.0 for cls_name in classes}

        # 5. Extract acoustic features for visual inspection
        acoustic_feats = audio.extract_acoustic_features(waveform, sr)

        return {
            "success": True,
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2) if confidence else None,
            "probabilities": probabilities,
            "classes": classes,
            "transformer_model": active_transformer,
            "classifier_type": package.get("classifier_type", "Logistic Regression"),
            "audio_duration": round(duration_sec, 2),
            "waveform": waveform,
            "sr": sr,
            "acoustic_features": acoustic_feats,
            "model_path": model_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }
