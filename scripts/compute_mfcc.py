import argparse
from pathlib import Path
import sys
import time

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path for `asr` package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr.features import compute_mfcc


def parse_args():
    p = argparse.ArgumentParser(description="Compute MFCCs for an audio file")
    p.add_argument("audio", type=Path, nargs="?", default=Path("data/raw/harvard.wav"),
                   help="Path to input audio file")
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    p.add_argument("--n-fft", type=int, default=400, help="FFT size")
    p.add_argument("--hop-length", type=int, default=160, help="Hop length")
    p.add_argument("--n-mels", type=int, default=80, help="Number of mel bands")
    p.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients")
    p.add_argument("--delta", action="store_true", help="Include delta features")
    p.add_argument("--delta-delta", action="store_true", help="Include delta-delta features")
    p.add_argument("--plot", action="store_true", help="Save a heatmap plot of MFCCs")
    p.add_argument("--out", type=Path, default=Path("outputs/mfcc_values.npy"), help="Output .npy path")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")

    print(f"Loading: {args.audio} -> sr={args.sr}")
    t0 = time.time()
    y, sr = librosa.load(args.audio, sr=args.sr, mono=True)
    print(f"Loaded waveform shape={y.shape} in {time.time()-t0:.2f}s")

    print("Computing MFCCs...")
    t1 = time.time()
    mfcc = compute_mfcc(
        y=y,
        sr=sr,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        d1=args.delta,
        d2=args.delta_delta,
    )
    dt = time.time()-t1
    print(f"MFCC shape={mfcc.shape} computed in {dt:.2f}s")

    args.out.parent.mkdir(exist_ok=True, parents=True)
    np.save(args.out, mfcc)
    print(f"Saved MFCC array to: {args.out}")

    if args.plot:
        plt.figure(figsize=(10, 4))
        im = plt.imshow(mfcc, aspect="auto", origin="lower", interpolation="nearest")
        plt.colorbar(im)
        plt.title("MFCCs")
        plt.xlabel("Frames")
        plt.ylabel("Coefficients")
        plot_path = args.out.with_suffix(".png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved MFCC plot to: {plot_path}")


if __name__ == "__main__":
    main()
