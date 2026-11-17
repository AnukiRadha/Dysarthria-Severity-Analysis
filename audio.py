"""
audio.py - Audio Preprocessing and Feature Extraction
-----------------------------------------------------
Handles audio loading, format conversion, 16 kHz resampling,
normalization, and acoustic visualization preparation.
"""

import io
import os
import numpy as np
import soundfile as sf
import librosa

TARGET_SAMPLING_RATE = 16000  # Standard native rate for Wav2Vec 2.0 and HuBERT
MAX_AUDIO_DURATION = 3.5      # Optimal seconds to capture dysarthric phonetics without CPU lag


def load_audio(source, target_sr=TARGET_SAMPLING_RATE, max_duration=MAX_AUDIO_DURATION):
    """
    Loads an audio file from a file path or in-memory byte buffer,
    converts it to mono float32, resamples to target_sr, and normalizes amplitude.
    
    Parameters:
    - source: file path string OR file-like byte buffer
    - target_sr: target sample rate (default: 16000 Hz)
    - max_duration: max duration in seconds
    
    Returns:
    - audio_waveform: 1D numpy array of float32 values
    - sr: target sample rate (16000)
    """
    try:
        # Check if source is a filepath or byte buffer
        if isinstance(source, (str, os.PathLike)):
            waveform, orig_sr = sf.read(source, dtype="float32")
        else:
            # BytesIO or UploadedFile
            if hasattr(source, "read"):
                source.seek(0)
                waveform, orig_sr = sf.read(source, dtype="float32")
            elif isinstance(source, bytes):
                buf = io.BytesIO(source)
                waveform, orig_sr = sf.read(buf, dtype="float32")
            else:
                raise ValueError("Unsupported audio source type.")

        # Convert stereo to mono if necessary
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample to 16 kHz if different
        if orig_sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

        # Truncate to max_duration if too long
        max_samples = int(max_duration * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Simple amplitude normalization (prevent clipping or vanishing signal)
        max_val = np.max(np.abs(waveform))
        if max_val > 1e-6:
            waveform = waveform / max_val

        return waveform.astype(np.float32), target_sr

    except Exception as e:
        # Fallback using librosa directly if soundfile encounters an odd header
        try:
            if isinstance(source, (str, os.PathLike)):
                waveform, orig_sr = librosa.load(source, sr=target_sr, mono=True, duration=max_duration)
                max_val = np.max(np.abs(waveform))
                if max_val > 1e-6:
                    waveform = waveform / max_val
                return waveform.astype(np.float32), target_sr
            else:
                raise e
        except Exception as fallback_error:
            raise RuntimeError(f"Error loading audio: {fallback_error}") from fallback_error


def get_audio_info(source):
    """
    Returns audio properties (duration, sampling rate, channels).
    Supports both file paths and file-like byte buffers.
    """
    try:
        if hasattr(source, "seek"):
            source.seek(0)
            info = sf.info(source)
            source.seek(0)
        else:
            info = sf.info(source)

        return {
            "duration": round(info.duration, 2),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype
        }
    except Exception:
        # Fallback reading with soundfile/librosa
        try:
            w, sr = load_audio(source)
            dur = len(w) / sr
            return {
                "duration": round(dur, 2),
                "samplerate": sr,
                "channels": 1,
                "format": "WAV",
                "subtype": "PCM_16"
            }
        except Exception:
            return {
                "duration": 0.0,
                "samplerate": 16000,
                "channels": 1,
                "format": "WAV",
                "subtype": "PCM_16"
            }


def extract_acoustic_features(waveform, sr=TARGET_SAMPLING_RATE):
    """
    Extracts classical acoustic features (MFCC, Zero-Crossing Rate, Spectral Centroid)
    useful for exploratory data analysis and audio visualization in Streamlit.
    """
    # 13 MFCCs
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    centroid_mean = float(np.mean(centroid))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    zcr_mean = float(np.mean(zcr))
    
    return {
        "mfcc": mfcc,
        "mfcc_mean": mfcc_mean,
        "spectral_centroid": centroid_mean,
        "zero_crossing_rate": zcr_mean
    }
