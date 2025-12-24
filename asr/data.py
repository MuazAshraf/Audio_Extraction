from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset as TorchDataset

from asr.tokenizer import CharTokenizer


def audio_bytes_to_waveform(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio bytes to mono waveform and sample rate."""
    with io.BytesIO(audio_bytes) as buf:
        waveform, sr = sf.read(buf, dtype="float32")
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform, sr


def waveform_to_mel(
    y: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
) -> torch.Tensor:
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]


class LibriShardDataset(TorchDataset):
    def __init__(self, hf_ds: Dataset, tokenizer: CharTokenizer, mel_cfg: dict):
        self.hf_ds = hf_ds.cast_column("audio", Audio(decode=False))
        self.tokenizer = tokenizer
        self.mel_cfg = mel_cfg

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        sample = self.hf_ds[idx]
        y, sr = audio_bytes_to_waveform(sample["audio"]["bytes"])
        mel = waveform_to_mel(
            y,
            sr,
            target_sr=self.mel_cfg["sr"],
            n_fft=self.mel_cfg["n_fft"],
            hop_length=self.mel_cfg["hop_length"],
            n_mels=self.mel_cfg["n_mels"],
        )
        token_ids = torch.tensor(self.tokenizer.encode(sample["text"]), dtype=torch.long)
        return mel, token_ids


def collate_batch(batch, pad_id: int, n_mels: int):
    mels, tokens = zip(*batch)
    max_t = max(m.shape[-1] for m in mels)
    max_len = max(t.size(0) for t in tokens)

    mel_batch = torch.zeros(len(batch), 1, n_mels, max_t, dtype=torch.float32)
    for i, m in enumerate(mels):
        mel_batch[i, :, :, : m.shape[-1]] = m

    token_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens):
        token_batch[i, : t.size(0)] = t

    return mel_batch, token_batch


@dataclass
class SplitData:
    train: LibriShardDataset
    val: LibriShardDataset
    test: LibriShardDataset


def prepare_data(
    source_path: Path,
    max_samples: int | None,
    train_ratio: float,
    val_ratio: float,
    mel_cfg: dict,
) -> tuple[SplitData, CharTokenizer]:
    if not source_path.exists():
        raise FileNotFoundError(f"Path not found: {source_path}")

    # Collect one or many arrow files
    if source_path.is_dir():
        arrow_files = sorted(source_path.glob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {source_path}")
    else:
        arrow_files = [source_path]

    datasets = [Dataset.from_file(str(p)) for p in arrow_files]
    hf_ds = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    hf_ds = hf_ds.shuffle(seed=42)
    if max_samples is not None:
        hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))

    texts = hf_ds["text"]
    tokenizer = CharTokenizer(texts)

    n = len(hf_ds)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ds = hf_ds.select(range(0, n_train))
    val_ds = hf_ds.select(range(n_train, n_train + n_val))
    test_ds = hf_ds.select(range(n_train + n_val, n))

    train = LibriShardDataset(train_ds, tokenizer, mel_cfg)
    val = LibriShardDataset(val_ds, tokenizer, mel_cfg)
    test = LibriShardDataset(test_ds, tokenizer, mel_cfg)

    return SplitData(train=train, val=val, test=test), tokenizer


def build_loaders(splits: SplitData, tokenizer: CharTokenizer, batch_size: int, n_mels: int):
    collate = lambda b: collate_batch(b, tokenizer.pad_id, n_mels)
    train_loader = DataLoader(splits.train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader
