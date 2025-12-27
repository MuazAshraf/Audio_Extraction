import os
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import Any, Dict, List
import librosa
import numpy as np
from jiwer import wer as compute_wer
import soundfile as sf
import io

from vocab import Vocabulary
from cnn_encoder import Seq2SeqASR

# Config
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0003  # Lower learning rate
CACHE_DIR = "/workspace/hf_cache_v2/downloads"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAD_CLIP = 5.0  # Gradient clipping

print(f"Using device: {DEVICE}")

# Vocabulary
vocab = Vocabulary()
print(f"Vocab size: {vocab.vocab_size}")


def find_parquet_files(cache_dir, split_name="train.100"):
    """Find all parquet files for a given split from the cached downloads."""
    parquet_files = []
    json_files = glob.glob(os.path.join(cache_dir, "*.json"))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            meta = json.load(f)
        url = meta.get("url", "")
        if split_name in url and url.endswith(".parquet"):
            # The actual parquet file has the same name without .json
            parquet_path = json_file.replace(".json", "")
            if os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
                parquet_files.append(parquet_path)

    return parquet_files


class LibriSpeechParquetDataset(Dataset):
    """Dataset that loads directly from parquet files."""

    def __init__(self, parquet_files, vocab):
        self.vocab = vocab
        self.data = []

        print(f"Loading {len(parquet_files)} parquet files...")
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                self.data.append({
                    "audio_bytes": row["audio"]["bytes"],
                    "text": row["text"]
                })
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Decode audio from bytes
        audio_bytes = item["audio_bytes"]
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample to 16kHz if needed
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        # Convert to float32
        audio_np = np.array(audio_data, dtype=np.float32)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio_np,
            sr=16000,
            n_mels=128,
            n_fft=400,
            hop_length=160
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Text to indices
        text = item["text"]
        labels = self.vocab.encode(text)

        return {
            "spectrogram": mel_db,
            "labels": labels,
            "text": text.lower()
        }


# Data Collator (padding)
@dataclass
class DataCollatorASR:
    def __call__(self, features: List[Dict[str, Any]]):
        specs = [torch.tensor(f["spectrogram"], dtype=torch.float32) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        texts = [f["text"] for f in features]

        # Pad spectrograms
        max_spec_len = max(s.size(1) for s in specs)
        padded_specs = []
        for s in specs:
            pad_len = max_spec_len - s.size(1)
            padded = torch.nn.functional.pad(s, (0, pad_len))
            padded_specs.append(padded)

        specs_batch = torch.stack(padded_specs).unsqueeze(1)  # (B, 1, 128, T)

        # Pad labels
        max_label_len = max(len(l) for l in labels)
        padded_labels = []
        for l in labels:
            pad_len = max_label_len - len(l)
            padded = torch.nn.functional.pad(l, (0, pad_len))
            padded_labels.append(padded)

        labels_batch = torch.stack(padded_labels)  # (B, T)

        return {
            "spectrogram": specs_batch,
            "labels": labels_batch,
            "texts": texts
        }


data_collator = DataCollatorASR()


# Load data from parquet files
print("Finding parquet files...")
parquet_files = find_parquet_files(CACHE_DIR, "train.100")
print(f"Found {len(parquet_files)} parquet files for train.100")

if len(parquet_files) == 0:
    raise RuntimeError("No parquet files found! Check CACHE_DIR path.")

# Create dataset
full_dataset = LibriSpeechParquetDataset(parquet_files, vocab)

# Split: 70% train, 20% val, 10% test
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator, num_workers=0)

# Model
model = Seq2SeqASR(vocab_size=vocab.vocab_size)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# Validation function
def validate(model, loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in loader:
            specs = batch["spectrogram"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            texts = batch["texts"]

            output = model(specs, labels, teacher_forcing=0.0)

            # Loss
            output_flat = output.reshape(-1, vocab.vocab_size)
            labels_flat = labels[:, 1:].reshape(-1)
            loss = criterion(output_flat, labels_flat)
            total_loss += loss.item()

            # Decode predictions for WER
            pred_indices = output.argmax(dim=2)
            for i in range(pred_indices.size(0)):
                pred_text = vocab.decode(pred_indices[i].tolist())
                all_preds.append(pred_text)
                all_refs.append(texts[i].lower())

    avg_loss = total_loss / len(loader)
    wer = compute_wer(all_refs, all_preds)

    return avg_loss, wer


# Training loop
print("Starting training...")
best_wer = float('inf')
start_epoch = 0

# NOTE: Model architecture changed - starting fresh training
# Old checkpoints are incompatible with new model (added batch norm + init_hidden layer)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    # Teacher forcing: start high (0.9) and decay
    teacher_forcing_ratio = max(0.5, 0.9 - epoch * 0.05)

    for batch_idx, batch in enumerate(train_loader):
        specs = batch["spectrogram"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Forward with teacher forcing
        output = model(specs, labels, teacher_forcing=teacher_forcing_ratio)

        # Loss
        output = output.reshape(-1, vocab.vocab_size)
        labels_loss = labels[:, 1:].reshape(-1)
        loss = criterion(output, labels_loss)

        # Backward with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, TF: {teacher_forcing_ratio:.2f}")

    # Training stats
    train_loss = total_loss / len(train_loader)

    # Validation after each epoch
    val_loss, val_wer = validate(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val WER: {val_wer:.4f}")

    # Learning rate scheduler
    scheduler.step(val_loss)

    # Save best model
    if val_wer < best_wer:
        best_wer = val_wer
        torch.save(model.state_dict(), "model_best.pt")
        print(f"  -> New best model saved! WER: {val_wer:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

# Final test evaluation
print("Evaluating on test set...")
test_loss, test_wer = validate(model, test_loader)
print(f"Test Loss: {test_loss:.4f} | Test WER: {test_wer:.4f}")

print("\nTraining complete!")
torch.save(model.state_dict(), "model_final.pt")
