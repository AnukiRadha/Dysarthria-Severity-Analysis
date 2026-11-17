"""
model.py - Pretrained Speech Transformers and Lightweight Classifiers
--------------------------------------------------------------------
Integrates actual Hugging Face Wav2Vec 2.0 and HuBERT speech transformers.
Keeps transformer models frozen and trains lightweight classifiers
(Logistic Regression, SVM) on the extracted 768-dimensional embeddings.
"""

import os
import joblib
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoFeatureExtractor,
    HubertModel
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import audio

# Pretrained model registry
MODEL_REGISTRY = {
    "Wav2Vec 2.0": {
        "hf_id": "facebook/wav2vec2-base",
        "description": "Meta Wav2Vec 2.0 Base (Pretrained on 960h LibriSpeech, 768-dim embeddings)",
        "embedding_dim": 768
    },
    "HuBERT": {
        "hf_id": "facebook/hubert-base-ls960",
        "description": "Meta HuBERT Base (Hidden-Unit BERT Pretrained on 960h LibriSpeech, 768-dim embeddings)",
        "embedding_dim": 768
    },
    "XLS-R": {
        "hf_id": "facebook/wav2vec2-xls-r-300m",
        "description": "Meta XLS-R Multilingual Speech Transformer (Cross-lingual 128 languages, 1024-dim embeddings)",
        "embedding_dim": 1024
    }
}

# In-memory model cache to avoid re-loading weights on every call
_LOADED_MODELS = {}


def get_available_models():
    """
    Returns the list of supported pretrained transformer model names.
    """
    return list(MODEL_REGISTRY.keys())


def load_transformer_model(model_name="Wav2Vec 2.0", device=None):
    """
    Loads and caches the requested pretrained speech transformer.
    Keeps weights frozen (eval mode, requires_grad=False).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {get_available_models()}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]["processor"], _LOADED_MODELS[model_name]["model"], device

    model_info = MODEL_REGISTRY[model_name]
    hf_id = model_info["hf_id"]

    try:
        if model_name in ["Wav2Vec 2.0", "XLS-R"]:
            processor = AutoProcessor.from_pretrained(hf_id, local_files_only=True)
            model = AutoModel.from_pretrained(hf_id, local_files_only=True)
        elif model_name == "HuBERT":
            processor = AutoFeatureExtractor.from_pretrained(hf_id, local_files_only=True)
            model = HubertModel.from_pretrained(hf_id, local_files_only=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception:
        # Fallback to online loading if offline cache lookup fails
        if model_name in ["Wav2Vec 2.0", "XLS-R"]:
            processor = AutoProcessor.from_pretrained(hf_id)
            model = AutoModel.from_pretrained(hf_id)
        else:
            processor = AutoFeatureExtractor.from_pretrained(hf_id)
            model = HubertModel.from_pretrained(hf_id)

    # Freeze transformer weights to reduce hardware requirements and compute time
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    _LOADED_MODELS[model_name] = {
        "processor": processor,
        "model": model,
        "device": device
    }

    return processor, model, device


def extract_embedding_from_waveform(waveform, model_name="Wav2Vec 2.0"):
    """
    Passes a 16 kHz audio waveform through the frozen transformer model
    and performs temporal mean-pooling to return a 768-dim numpy feature vector.
    """
    processor, model, device = load_transformer_model(model_name)

    # Process waveform into tensor
    inputs = processor(
        waveform,
        sampling_rate=audio.TARGET_SAMPLING_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        # outputs.last_hidden_state has shape (batch_size, time_frames, hidden_dim)
        hidden_states = outputs.last_hidden_state
        # Mean pooling across the time dimension
        embedding = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()

    return embedding.astype(np.float32)


def extract_embedding_from_file(file_path, model_name="Wav2Vec 2.0"):
    """
    Helper to extract transformer embedding directly from an audio file.
    """
    waveform, _ = audio.load_audio(file_path)
    return extract_embedding_from_waveform(waveform, model_name=model_name)


def extract_dataset_embeddings(df, model_name="Wav2Vec 2.0", batch_size=8, cache_path=None, progress_callback=None):
    """
    Extracts embeddings for a DataFrame of audio recordings using batched processing
    for significant speedup on CPU. Supports optional disk caching.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_paths = data.get("file_paths", [])
            current_paths = df["file_path"].values
            if len(cached_paths) == len(current_paths) and np.array_equal(cached_paths, current_paths):
                return data["embeddings"]
        except Exception:
            pass

    processor, transformer, device = load_transformer_model(model_name)
    all_embeddings = []
    file_paths = df["file_path"].tolist()
    total_files = len(file_paths)

    for i in range(0, total_files, batch_size):
        batch_paths = file_paths[i:i + batch_size]
        waveforms = []
        for p in batch_paths:
            try:
                w, _ = audio.load_audio(p)
                if len(w) < 160:  # Minimum 10ms
                    w = np.pad(w, (0, 160 - len(w)))
                waveforms.append(w)
            except Exception:
                waveforms.append(np.zeros(1600, dtype=np.float32))

        # Pad and tokenize batch
        inputs = processor(
            waveforms,
            sampling_rate=audio.TARGET_SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(device)
        attn_mask = getattr(inputs, "attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            outputs = transformer(input_values)
            hidden_states = outputs.last_hidden_state  # shape: (B, T, D)
            
            # Temporal mean pooling
            if attn_mask is not None:
                # Downsample attention mask to match transformer hidden states temporal length
                # Wav2Vec2/HuBERT downsamples raw audio by a factor of ~320
                ratio = input_values.shape[1] / hidden_states.shape[1]
                ds_mask = torch.nn.functional.interpolate(
                    attn_mask.unsqueeze(1).float(),
                    size=hidden_states.shape[1],
                    mode="nearest"
                ).squeeze(1)
                mask_expanded = ds_mask.unsqueeze(-1)
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                batch_embeddings = torch.mean(hidden_states, dim=1).cpu().numpy()

        all_embeddings.append(batch_embeddings)

        if progress_callback:
            progress_callback(min(i + batch_size, total_files), total_files)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            file_paths=df["file_path"].values,
            labels=df["label"].values if "label" in df.columns else []
        )

    return embeddings


def build_classifier(classifier_type="Logistic Regression"):
    """
    Builds a lightweight scikit-learn classifier pipeline with standard scaling.
    """
    if classifier_type == "SVM":
        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    elif classifier_type == "Linear SVM":
        clf = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    else:
        # Default: Logistic Regression
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    return pipeline


def save_trained_model(pipeline, label_encoder, model_name, classifier_type, metrics, save_dir="saved_models"):
    """
    Saves the trained classifier, label encoder, and training metadata to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name.lower().replace(' ', '_')}_{classifier_type.lower().replace(' ', '_')}.joblib"
    file_path = os.path.join(save_dir, filename)

    package = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "model_name": model_name,
        "classifier_type": classifier_type,
        "classes": list(label_encoder.classes_),
        "metrics": metrics
    }
    joblib.dump(package, file_path)
    return file_path


def load_trained_model(file_path):
    """
    Loads a saved trained classifier package.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return joblib.load(file_path)
