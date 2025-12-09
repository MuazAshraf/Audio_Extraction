# import os
# from pathlib import Path

# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import time

# # Use a non-interactive backend to avoid GUI blocking
# plt.switch_backend("Agg")


# # 1. Config
# AUDIO_PATH = Path("data/raw/harvard.wav")  # change if your file has a different name
# TARGET_SR = 16000                          # we'll resample to 16kHz
# N_FFT = 400                                # ~25 ms window at 16kHz
# HOP_LENGTH = 160                           # ~10 ms hop
# N_MELS = 80                                # number of mel bands


# def main():
#     if not AUDIO_PATH.exists():
#         raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

#     # 2. Load audio with librosa (mono, resampled)
#     t0 = time.time()
#     print("Loading audio...", flush=True)
#     y, sr = librosa.load(AUDIO_PATH, sr=TARGET_SR, mono=True)
#     print(f"Loaded audio: {AUDIO_PATH}, waveform shape={y.shape}, sr={sr} (t={time.time()-t0:.2f}s)", flush=True)

#     # 3. Plot waveform
#     print("Plotting waveform...", flush=True)
#     plt.figure(figsize=(10, 3))
#     librosa.display.waveshow(y, sr=sr)
#     plt.title("Waveform")
#     plt.xlabel("Time (s)")
#     plt.tight_layout()
#     os.makedirs("outputs", exist_ok=True)
#     plt.savefig("outputs/waveform.png")
#     plt.close()
#     print("Saved waveform plot to outputs/waveform.png", flush=True)

#     # 4. Compute STFT magnitude spectrogram
#     print("Computing STFT...", flush=True)
#     D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)   # complex-valued
#     S_mag = np.abs(D)                                         # magnitude
#     S_db = librosa.amplitude_to_db(S_mag, ref=np.max)         # to dB scale

#     print("Rendering linear spectrogram...", flush=True)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(
#         S_db,
#         sr=sr,
#         hop_length=HOP_LENGTH,
#         x_axis="time",
#         y_axis="linear"
#     )
#     plt.colorbar(format="%+2.0f dB")
#     plt.title("Spectrogram (linear freq, dB)")
#     plt.tight_layout()
#     plt.savefig("outputs/spectrogram_linear.png")
#     plt.close()
#     print("Saved linear spectrogram to outputs/spectrogram_linear.png", flush=True)

#     # 5. Compute log-Mel spectrogram
#     print("Computing mel spectrogram...", flush=True)
#     S_mel = librosa.feature.melspectrogram(
#         y=y,
#         sr=sr,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS,
#         power=2.0   # power spectrogram
#     )
#     S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

#     print("Rendering log-mel spectrogram...", flush=True)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(
#         S_mel_db,
#         sr=sr,
#         hop_length=HOP_LENGTH,
#         x_axis="time",
#         y_axis="mel"
#     )
#     plt.colorbar(format="%+2.0f dB")
#     plt.title("Log-Mel Spectrogram")
#     plt.tight_layout()
#     plt.savefig("outputs/spectrogram_logmel.png")
#     plt.close()
#     print("Saved log-mel spectrogram to outputs/spectrogram_logmel.png", flush=True)


# if __name__ == "__main__":
#     main()

import os
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# from xgboost import XGBClassifier  # pip install xgboost

# Use a non-interactive backend to avoid GUI blocking
plt.switch_backend("Agg")


# 1. Config
AUDIO_PATH = Path("data/raw/harvard.wav")  # for visualization (single file)
TARGET_SR = 16000                          # we'll resample to 16kHz
N_FFT = 400                                # ~25 ms window at 16kHz
HOP_LENGTH = 160                           # ~10 ms hop
N_MELS = 80                                # number of mel bands
N_MFCC = 13                                # number of MFCC coefficients

# For classification: expects data/raw/<class_name>/*.wav
DATA_ROOT = Path("data/raw")


# ---------- MFCC FEATURE EXTRACTION ----------

def compute_mfcc(y, sr):
    """Compute MFCC matrix [n_mfcc, n_frames] from waveform."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS  # uses mel filterbank internally
    )
    return mfcc


def summarize_mfcc(mfcc):
    """
    Summarize MFCC matrix [n_mfcc, n_frames] into a 1D feature vector.

    Here we use:
      - mean over time
      - std over time 

    => feature length = 2 * N_MFCC
    """
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std], axis=0)
    return feat  # shape: [2 * N_MFCC]


def extract_mfcc_feature_from_file(audio_path: Path):
    """Load an audio file and return MFCC feature vector [2 * N_MFCC]."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    # Optional: trim to max 5 seconds
    max_seconds = 5
    max_samples = TARGET_SR * max_seconds
    if y.shape[0] > max_samples:
        y = y[:max_samples]

    mfcc = compute_mfcc(y, sr)
    feat = summarize_mfcc(mfcc)
    return feat


def build_mfcc_dataset(data_root: Path):
    """
    Build dataset X, y using MFCC summary features.

    Expects:
        data/raw/
          class_name_1/
            *.wav / *.flac / *.mp3
          class_name_2/
            ...

    Returns:
        X: [num_samples, 2 * N_MFCC]
        y: [num_samples]
    """
    X = []
    y = []

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root.resolve()}")

    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        # skip files directly under data/raw (like harvard.wav)
        if label.endswith(".wav") or label.endswith(".mp3") or label.endswith(".flac"):
            continue

        print(f"Processing class: {label}", flush=True)

        audio_files = []
        audio_files += list(class_dir.rglob("*.wav"))
        audio_files += list(class_dir.rglob("*.flac"))
        audio_files += list(class_dir.rglob("*.mp3"))

        if not audio_files:
            print(f"  WARNING: no audio files in {class_dir}", flush=True)
            continue

        for audio_path in audio_files:
            try:
                feat = extract_mfcc_feature_from_file(audio_path)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"  Error with {audio_path}: {e}", flush=True)

    if len(X) == 0:
        return np.array([]), np.array([])

    X = np.array(X)
    y = np.array(y)
    return X, y


# ---------- TRAINING: LR, SVM, DT, RF, XGBoost ----------

def train_all_models_on_mfcc(X, y):
    """Train LR, SVM, DT, RF, XGBoost on MFCC features and print accuracy & F1."""
    if X.size == 0:
        print("No data available for training (X is empty).", flush=True)
        return

    unique_labels = np.unique(y)
    print("Classes:", unique_labels, flush=True)
    if len(unique_labels) < 2:
        print("Need at least 2 classes for classification.", flush=True)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf")
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=None,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        # "XGBoost": XGBClassifier(
        #     n_estimators=200,
        #     learning_rate=0.1,
        #     max_depth=4,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     objective="multi:softprob",
        #     eval_metric="mlogloss",
        #     random_state=42
        # )
    }

    for name, model in models.items():
        print("\n==============================", flush=True)
        print(f"Training model: {name}", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy:      {acc:.4f}", flush=True)
        print(f"F1 (macro):    {f1_macro:.4f}", flush=True)
        print(f"F1 (weighted): {f1_weighted:.4f}", flush=True)
        print("\nClassification report:", flush=True)
        print(classification_report(y_test, y_pred), flush=True)


# ---------- VISUALIZATION (single harvard.wav) ----------

def main():
    # ========== PART 1: VISUALIZATION (single file) ==========
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    t0 = time.time()
    print("Loading audio...", flush=True)
    y, sr = librosa.load(AUDIO_PATH, sr=TARGET_SR, mono=True)
    print(
        f"Loaded audio: {AUDIO_PATH}, waveform shape={y.shape}, sr={sr} "
        f"(t={time.time()-t0:.2f}s)",
        flush=True,
    )

    print("Plotting waveform...", flush=True)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/waveform.png")
    plt.close()
    print("Saved waveform plot to outputs/waveform.png", flush=True)

    print("Computing STFT...", flush=True)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)   # complex-valued
    S_mag = np.abs(D)                                         # magnitude
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)         # to dB scale

    print("Rendering linear spectrogram...", flush=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (linear freq, dB)")
    plt.tight_layout()
    plt.savefig("outputs/spectrogram_linear.png")
    plt.close()
    print("Saved linear spectrogram to outputs/spectrogram_linear.png", flush=True)

    print("Computing mel spectrogram...", flush=True)
    S_mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0   # power spectrogram
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    print("Rendering log-mel spectrogram...", flush=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.savefig("outputs/spectrogram_logmel.png")
    plt.close()
    print("Saved log-mel spectrogram to outputs/spectrogram_logmel.png", flush=True)

    print("Computing MFCCs (single file)...", flush=True)
    mfcc_single = compute_mfcc(y, sr)
    print(f"Single-file MFCC shape: {mfcc_single.shape}", flush=True)

    print("Rendering MFCCs for single file...", flush=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfcc_single,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time"
    )
    plt.colorbar(format="%+2.0f")
    plt.title(f"MFCCs ({N_MFCC} coefficients)")
    plt.ylabel("MFCC index")
    plt.tight_layout()
    plt.savefig("outputs/mfcc_single.png")
    plt.close()
    print("Saved MFCC plot for single file to outputs/mfcc_single.png", flush=True)

    # ========== PART 2: CLASSIFICATION WITH MFCC FEATURES ==========
    print("\nBuilding MFCC dataset from:", DATA_ROOT, flush=True)
    X, y_labels = build_mfcc_dataset(DATA_ROOT)
    print("X shape:", X.shape, "y shape:", y_labels.shape, flush=True)

    if X.size > 0:
        print("\nTraining LR, SVM, DT, RF, XGBoost on MFCC features...", flush=True)
        train_all_models_on_mfcc(X, y_labels)
    else:
        print("No data found in data/raw/<class_name>/; skipping classification.", flush=True)


if __name__ == "__main__":
    main()
