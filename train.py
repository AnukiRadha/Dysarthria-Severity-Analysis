"""
train.py - Speaker-Independent Model Training and Evaluation Pipeline
---------------------------------------------------------------------
Executes the research training pipeline:
1. Filters dataset and prepares labels.
2. Applies strict speaker-independent splitting (GroupShuffleSplit).
3. Extracts representations via frozen Wav2Vec 2.0 or HuBERT.
4. Trains a lightweight classifier (Logistic Regression or SVM).
5. Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
6. Persists models and benchmark comparisons for the dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import data
import model


COMPARISON_FILE = "results/model_comparisons.json"


def update_comparison_registry(model_name, classifier_type, task_type, metrics):
    """
    Saves and updates real benchmark comparison records to JSON.
    Never uses fake data; only actual evaluation runs are recorded.
    """
    os.makedirs(os.path.dirname(COMPARISON_FILE), exist_ok=True)
    records = []

    if os.path.exists(COMPARISON_FILE):
        try:
            with open(COMPARISON_FILE, "r") as f:
                records = json.load(f)
        except Exception:
            records = []

    entry_key = f"{model_name} + {classifier_type} ({task_type})"
    updated = False

    new_record = {
        "key": entry_key,
        "model_name": model_name,
        "classifier_type": classifier_type,
        "task_type": task_type,
        "accuracy": round(float(metrics["accuracy"]), 4),
        "precision_macro": round(float(metrics["precision_macro"]), 4),
        "recall_macro": round(float(metrics["recall_macro"]), 4),
        "f1_macro": round(float(metrics["f1_macro"]), 4),
        "f1_weighted": round(float(metrics["f1_weighted"]), 4),
        "train_samples": int(metrics["train_samples"]),
        "test_samples": int(metrics["test_samples"]),
        "train_speakers": metrics["train_speakers"],
        "test_speakers": metrics["test_speakers"],
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for i, r in enumerate(records):
        if r.get("key") == entry_key:
            records[i] = new_record
            updated = True
            break

    if not updated:
        records.append(new_record)

    with open(COMPARISON_FILE, "w") as f:
        json.dump(records, f, indent=2)

    return records


def get_comparison_registry():
    """
    Loads all recorded benchmark runs.
    """
    if os.path.exists(COMPARISON_FILE):
        try:
            with open(COMPARISON_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def run_training_pipeline(
    model_name="Wav2Vec 2.0",
    classifier_type="Logistic Regression",
    task_type="4-Class Severity",
    mic_filter="All",
    max_samples_per_speaker=15,
    test_size=0.25,
    batch_size=8,
    random_state=42,
    progress_callback=None
):
    """
    Runs the complete speaker-independent training & evaluation flow.
    
    Returns a dictionary of metrics, confusion matrix, and split details.
    """
    if progress_callback:
        progress_callback("Scanning dataset and applying filters...", 0.05)

    df_raw = data.scan_dataset()
    df = data.filter_dataset(
        df_raw,
        task_type=task_type,
        mic_filter=mic_filter,
        max_samples_per_speaker=max_samples_per_speaker,
        random_state=random_state
    )

    if progress_callback:
        progress_callback("Performing speaker-independent split...", 0.15)

    train_df, test_df = data.speaker_independent_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    train_speakers = sorted(train_df["speaker_id"].unique().tolist())
    test_speakers = sorted(test_df["speaker_id"].unique().tolist())

    # Encode class labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    
    # Check if test set contains any unseen class labels
    # If a class is missing from test due to random split, warn or handle gracefully
    unseen_classes = set(test_df["label"]) - set(label_encoder.classes_)
    if unseen_classes:
        # Fallback filter unseen test instances
        test_df = test_df[test_df["label"].isin(label_encoder.classes_)]
    
    y_test = label_encoder.transform(test_df["label"])
    class_names = list(label_encoder.classes_)

    if progress_callback:
        progress_callback(f"Extracting {model_name} embeddings for training set ({len(train_df)} samples)...", 0.25)

    # Cache key for training embeddings
    clean_model_tag = model_name.lower().replace(" ", "_")
    cache_train = f"results/emb_{clean_model_tag}_{task_type[:4].lower()}_train_{len(train_df)}.npz"
    cache_test = f"results/emb_{clean_model_tag}_{task_type[:4].lower()}_test_{len(test_df)}.npz"

    def train_progress(curr, tot):
        if progress_callback:
            frac = 0.25 + 0.40 * (curr / max(1, tot))
            progress_callback(f"Extracting {model_name} train features: {curr}/{tot}", frac)

    X_train = model.extract_dataset_embeddings(
        train_df,
        model_name=model_name,
        batch_size=batch_size,
        cache_path=cache_train,
        progress_callback=train_progress
    )

    if progress_callback:
        progress_callback(f"Extracting {model_name} embeddings for test set ({len(test_df)} samples)...", 0.65)

    def test_progress(curr, tot):
        if progress_callback:
            frac = 0.65 + 0.20 * (curr / max(1, tot))
            progress_callback(f"Extracting {model_name} test features: {curr}/{tot}", frac)

    X_test = model.extract_dataset_embeddings(
        test_df,
        model_name=model_name,
        batch_size=batch_size,
        cache_path=cache_test,
        progress_callback=test_progress
    )

    if progress_callback:
        progress_callback(f"Fitting {classifier_type} classifier on {model_name} embeddings...", 0.88)

    # Train classifier pipeline
    pipeline = model.build_classifier(classifier_type=classifier_type)
    pipeline.fit(X_train, y_train)

    if progress_callback:
        progress_callback("Evaluating model performance on unseen speakers...", 0.95)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    clf_report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    results = {
        "model_name": model_name,
        "classifier_type": classifier_type,
        "task_type": task_type,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_speakers": train_speakers,
        "test_speakers": test_speakers,
        "classes": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report_dict,
        "classification_report_text": clf_report_text
    }

    # Save model artifact
    saved_path = model.save_trained_model(
        pipeline=pipeline,
        label_encoder=label_encoder,
        model_name=model_name,
        classifier_type=classifier_type,
        metrics=results
    )
    results["saved_model_path"] = saved_path

    # Save specific metrics JSON
    metrics_file = f"results/metrics_{clean_model_tag}.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    # Update comparison registry
    update_comparison_registry(model_name, classifier_type, task_type, results)

    if progress_callback:
        progress_callback("Training completed successfully!", 1.0)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Training Dysarthria Severity Classifier (Speaker-Independent)")
    print("=" * 60)
    res = run_training_pipeline(
        model_name="Wav2Vec 2.0",
        classifier_type="Logistic Regression",
        task_type="4-Class Severity",
        max_samples_per_speaker=30
    )
    print("\nTraining Results Summary:")
    print(f"Accuracy: {res['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {res['f1_macro']:.4f}")
    print(f"Train Speakers ({len(res['train_speakers'])}): {res['train_speakers']}")
    print(f"Test Speakers ({len(res['test_speakers'])}): {res['test_speakers']}")
    print("\nClassification Report:\n", res["classification_report_text"])
