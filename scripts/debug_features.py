import numpy as np
import librosa
from pathlib import Path

# Config
AUDIO_PATH = Path("data/raw/harvard.wav")  # adjust if needed
TARGET_SR = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_MFCC = 13

def main():
    print("=== debug_features.py ===")
    print(f"Looking for audio at: {AUDIO_PATH.resolve()}")

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    print("Loading audio...")
    y, sr = librosa.load(AUDIO_PATH, sr=TARGET_SR, mono=True)
    print(f"Loaded: sr={sr}, waveform shape={y.shape}")

    # Optional: trim to first 5 seconds
    max_seconds = 5
    max_samples = TARGET_SR * max_seconds
    if y.shape[0] > max_samples:
        y = y[:max_samples]
        print(f"Trimmed to {max_seconds}s, new shape={y.shape}")

    # Log-Mel
    print("Computing log-Mel...")
    S_mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    print("Log-Mel shape:", S_mel_db.shape)  # (n_mels, n_frames)

    # MFCC
    print("Computing MFCCs...")
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    print("MFCC shape:", mfcc.shape)  # (n_mfcc, n_frames)

    # Save arrays for later, if needed
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "logmel_values.npy", S_mel_db)
    np.save(out_dir / "mfcc_values.npy", mfcc)
    print("Saved logmel_values.npy and mfcc_values.npy in outputs/")

    print("=== DONE debug_features.py ===")

if __name__ == "__main__":
    main()
