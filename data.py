"""
data.py - Dataset Handling and Speaker-Independent Splitting
------------------------------------------------------------
This module inspects the TORGO dysarthric speech dataset, builds a clean
metadata DataFrame, and creates speaker-independent train/test splits.
"""

import os
import glob
import wave
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Standard TORGO clinical severity classification based on research literature
# (Rudzicz et al., Language Resources & Evaluation)
SPEAKER_METADATA = {
    # Control (Healthy) Speakers
    "FC01": {"gender": "Female", "dysarthric": False, "severity": "Control"},
    "FC02": {"gender": "Female", "dysarthric": False, "severity": "Control"},
    "FC03": {"gender": "Female", "dysarthric": False, "severity": "Control"},
    "MC01": {"gender": "Male",   "dysarthric": False, "severity": "Control"},
    "MC02": {"gender": "Male",   "dysarthric": False, "severity": "Control"},
    "MC03": {"gender": "Male",   "dysarthric": False, "severity": "Control"},
    "MC04": {"gender": "Male",   "dysarthric": False, "severity": "Control"},
    # Dysarthric Speakers
    "F01":  {"gender": "Female", "dysarthric": True,  "severity": "Severe"},
    "F03":  {"gender": "Female", "dysarthric": True,  "severity": "Moderate"},
    "F04":  {"gender": "Female", "dysarthric": True,  "severity": "Mild"},
    "M01":  {"gender": "Male",   "dysarthric": True,  "severity": "Severe"},
    "M02":  {"gender": "Male",   "dysarthric": True,  "severity": "Severe"},
    "M03":  {"gender": "Male",   "dysarthric": True,  "severity": "Mild"},
    "M04":  {"gender": "Male",   "dysarthric": True,  "severity": "Severe"},
    "M05":  {"gender": "Male",   "dysarthric": True,  "severity": "Moderate"}, # Clinically Moderate-to-Severe
}

ALL_SPEAKER_IDS = list(SPEAKER_METADATA.keys())


def find_dataset_root(possible_dirs=None):
    """
    Locates the active dataset directory ('dataset' or 'archive').
    """
    if possible_dirs is None:
        possible_dirs = ["dataset", "archive", "../archive", "../dataset"]
    
    for d in possible_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            # Verify it contains at least one of the expected subfolders
            sub = os.listdir(d)
            if any(k in sub for k in ["F_Con", "F_Dys", "M_Con", "M_Dys"]):
                return d
    return "archive"


def extract_speaker_from_path(file_path):
    """
    Identifies the speaker ID from the directory path or file name.
    Example: 'wav_headMic_FC01S01_0001.wav' -> 'FC01'
    """
    for spk in ALL_SPEAKER_IDS:
        # Match speaker token surrounded by underscores or at end of session name
        if f"_{spk}" in file_path or f"{spk}S" in file_path or file_path.endswith(spk):
            return spk
    # Fallback search
    for spk in ALL_SPEAKER_IDS:
        if spk in file_path:
            return spk
    return "Unknown"


def scan_dataset(dataset_dir=None, cache_csv="results/dataset_metadata.csv"):
    """
    Scans all audio files, checks validity, and extracts speaker/label metadata.
    Results are cached to CSV for high performance in the Streamlit app.
    """
    if dataset_dir is None:
        dataset_dir = find_dataset_root()

    # Use cached CSV if available and up-to-date
    if os.path.exists(cache_csv):
        try:
            df = pd.read_csv(cache_csv)
            if len(df) > 0 and os.path.exists(df.iloc[0]["file_path"]):
                return df
        except Exception:
            pass

    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    records = []

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                full_path = os.path.join(root, f)
                speaker_id = extract_speaker_from_path(full_path)
                
                # Check mic type
                mic_type = "headMic" if "headmic" in full_path.lower() else "arrayMic"
                
                # Retrieve clinical info
                info = SPEAKER_METADATA.get(speaker_id, {
                    "gender": "Unknown",
                    "dysarthric": False,
                    "severity": "Unknown"
                })
                
                # Quick verification of audio header
                try:
                    file_size = os.path.getsize(full_path)
                    is_valid = file_size > 44  # Valid WAV header is at least 44 bytes
                except Exception:
                    is_valid = False

                if is_valid:
                    records.append({
                        "file_path": full_path,
                        "file_name": f,
                        "speaker_id": speaker_id,
                        "gender": info["gender"],
                        "dysarthric": info["dysarthric"],
                        "severity": info["severity"],
                        "mic_type": mic_type,
                        "language": "English (TORGO Corpus)",
                        "file_size_bytes": file_size
                    })

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(cache_csv, index=False)
    return df


def get_dataset_summary(df=None):
    """
    Returns high-level statistics for display on the dashboard.
    """
    if df is None:
        df = scan_dataset()

    total_samples = len(df)
    unique_speakers = df["speaker_id"].nunique()
    languages = df["language"].unique().tolist() if "language" in df.columns else ["English (TORGO)"]
    severity_counts = df["severity"].value_counts().to_dict()
    gender_counts = df["gender"].value_counts().to_dict()
    mic_counts = df["mic_type"].value_counts().to_dict()

    return {
        "total_samples": total_samples,
        "unique_speakers": unique_speakers,
        "speaker_list": sorted(df["speaker_id"].unique().tolist()),
        "languages": languages,
        "severity_counts": severity_counts,
        "gender_counts": gender_counts,
        "mic_counts": mic_counts
    }


def filter_dataset(df, task_type="4-Class Severity", mic_filter="All", max_samples_per_speaker=None, random_state=42):
    """
    Prepares a clean subset of the dataset for training or evaluation.
    
    Parameters:
    - task_type: '4-Class Severity' (Control, Mild, Moderate, Severe) or 'Binary' (Control, Dysarthric)
    - mic_filter: 'All', 'headMic' (cleaner speech), or 'arrayMic'
    - max_samples_per_speaker: limit samples per speaker to balance data and reduce CPU load
    """
    filtered = df.copy()

    # Filter out unknown speakers or labels
    filtered = filtered[filtered["severity"] != "Unknown"]

    # Filter by microphone type if requested
    if mic_filter in ["headMic", "arrayMic"]:
        filtered = filtered[filtered["mic_type"] == mic_filter]

    # Assign target label
    if task_type == "Binary":
        filtered["label"] = filtered["dysarthric"].apply(lambda x: "Dysarthric" if x else "Control")
    else:
        filtered["label"] = filtered["severity"]

    # Sample uniformly per speaker if specified (crucial for CPU training speed & balance)
    if max_samples_per_speaker is not None and max_samples_per_speaker > 0:
        sampled_list = []
        for spk, group in filtered.groupby("speaker_id"):
            if len(group) > max_samples_per_speaker:
                sampled_list.append(group.sample(n=max_samples_per_speaker, random_state=random_state))
            else:
                sampled_list.append(group)
        filtered = pd.concat(sampled_list, ignore_index=True)

    return filtered.reset_index(drop=True)


def speaker_independent_split(df, test_size=0.25, random_state=42):
    """
    Performs a strict speaker-independent train/test split using GroupShuffleSplit.
    
    Guarantees that NO speaker present in the training set ever appears
    in the testing set.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["speaker_id"]
    
    train_idx, test_idx = next(gss.split(df, groups=groups))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    # Verification check: Ensure zero overlap between train and test speakers
    train_speakers = set(train_df["speaker_id"].unique())
    test_speakers = set(test_df["speaker_id"].unique())
    overlap = train_speakers.intersection(test_speakers)
    
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Speaker leakage detected! Overlapping speakers: {overlap}")
        
    return train_df, test_df
