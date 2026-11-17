# Dysarthria Severity Classification Using Pretrained Speech Transformers

A research project evaluating speech representations from self-supervised speech transformers (**Wav2Vec 2.0** and **HuBERT**) for automated dysarthria severity classification under a strict **speaker-independent evaluation protocol**.

---

## 📌 Project Architecture

```
dysarthria_project/
│
├── app.py              # Modern Streamlit research dashboard (5 pages)
├── data.py             # Dataset loader, metadata scanner, and speaker-independent splitter
├── audio.py            # Audio preprocessing, 16 kHz mono conversion, acoustic analysis
├── model.py            # Pretrained Wav2Vec 2.0 & HuBERT transformers + lightweight classifiers
├── train.py            # Speaker-independent training and evaluation pipeline
├── predict.py          # Inference engine for uploaded/recorded speech
├── requirements.txt    # Compatible dependencies
│
├── dataset/            # Symlinked to existing TORGO dataset (archive/)
├── saved_models/       # Persisted trained classifiers (.joblib)
└── results/            # Evaluation metrics, confusion matrices, and benchmark comparisons
```

---

## 🔬 Dataset Overview (TORGO Corpus)

- **Total Audio Recordings:** 17,632 valid utterances
- **Format:** 16,000 Hz, 16-bit PCM Mono WAV
- **Speakers:** 15 total (8 dysarthric, 7 healthy control)
- **Languages:** English (North American)
- **Indian Languages:** None (TORGO was recorded at the University of Toronto & Holland Bloorview Kids Rehabilitation Hospital)

### Speaker & Clinical Severity Breakdown

| Severity Class | Speakers | Number of Utterances |
| :--- | :--- | :--- |
| **Control** | FC01, FC02, FC03, MC01, MC02, MC03, MC04 | 11,455 |
| **Mild** | F04, M03 | 1,535 |
| **Moderate** | F03, M05 (Moderate-to-Severe) | 2,076 |
| **Severe** | F01, M01, M02, M04 | 2,566 |

---

## 🛡️ Speaker-Independent Protocol (Zero Leakage)

To guarantee that the model learns pathology markers rather than memorizing individual speaker vocal acoustics:
- The dataset is split using `GroupShuffleSplit` on `speaker_id`.
- **Zero Overlap:** No speaker appearing in the training set ever appears in the test set.

---

## 🚀 How to Run the Project

### 1. Launch the Streamlit Dashboard

Run the following command in your terminal:

```bash
.venv/bin/streamlit run app.py
```

Or, if your virtual environment is activated (`source .venv/bin/activate`):

```bash
streamlit run app.py
```

Then open your browser at: **`http://localhost:8501`**

---

### 2. Run Speaker-Independent Training via CLI (Optional)

To train or re-evaluate the models from the terminal:

```bash
# Train Wav2Vec 2.0
.venv/bin/python train.py
```

---

### 3. Run Inference via CLI (Optional)

```bash
.venv/bin/python -c "
import predict
result = predict.predict_speech('dataset/F_Dys/wav_headMic_F01/wav_headMic_F01_0001.wav')
print('Predicted Severity:', result['prediction'])
print('Confidence:', result['confidence'], '%')
print('Probabilities:', result['probabilities'])
"
```

---

## 📊 Benchmark Results

| Model Architecture | Downstream Classifier | Unseen Test Accuracy | Weighted F1 | Unseen Test Speakers |
| :--- | :--- | :--- | :--- | :--- |
| **Wav2Vec 2.0** (`facebook/wav2vec2-base`) | Logistic Regression | **45.0%** | **0.500** | F01, M04, MC01, MC03 |
| **HuBERT** (`facebook/hubert-base-ls960`) | Logistic Regression | **65.0%** | **0.750** | F01, M04, MC01, MC03 |
