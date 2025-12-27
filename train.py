import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_AUDIO_DECODER"] = "soundfile"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List
import librosa
import numpy as np
import evaluate

from vocab import Vocabulary
from cnn_encoder import Seq2SeqASR
# Config
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Vocabulary
vocab = Vocabulary()
print(f"Vocab size: {vocab.vocab_size}")

# WER Metric (using evaluate library)
wer_metric = evaluate.load("wer")


# Preprocessing function
def prepare_dataset(batch):
    audio = batch["audio"]
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=np.array(audio["array"], dtype=np.float32),
        sr=16000,
        n_mels=128,
        n_fft=400,
        hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    batch["spectrogram"] = mel_db
    
    # Text to indices
    batch["labels"] = vocab.encode(batch["text"])
    
    return batch


# Data Collator (padding)
@dataclass
class DataCollatorASR:
    def __call__(self, features: List[Dict[str, Any]]):
        # Get spectrograms and labels
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


# Load and split data (70/20/10)
print("Loading dataset...")
dataset = load_dataset("SPRINGLab/LibriSpeech-100", split='train')

# Split: 70% train, 30% temp
train_temp = dataset.train_test_split(test_size=0.30, seed=42)
train_data = train_temp["train"]

# Split temp: 20% val, 10% test
val_test = train_temp["test"].train_test_split(test_size=0.333, seed=42)
val_data = val_test["train"]
test_data = val_test["test"]

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# Cast audio column and preprocess
train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
val_data = val_data.cast_column("audio", Audio(sampling_rate=16000))
test_data = test_data.cast_column("audio", Audio(sampling_rate=16000))

train_data = train_data.map(prepare_dataset, remove_columns=["audio", "speaker", "id"])
val_data = val_data.map(prepare_dataset, remove_columns=["audio", "speaker", "id"])
test_data = test_data.map(prepare_dataset, remove_columns=["audio", "speaker", "id"])

# DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

# Model
model = Seq2SeqASR(vocab_size=vocab.vocab_size)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


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
    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    
    return avg_loss, wer


# Training loop
print("Starting training...")
best_wer = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        specs = batch["spectrogram"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        # Forward
        output = model(specs, labels)
        
        # Loss
        output = output.reshape(-1, vocab.vocab_size)
        labels_loss = labels[:, 1:].reshape(-1)
        loss = criterion(output, labels_loss)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Training stats
    train_loss = total_loss / len(train_loader)
    
    # Validation after each epoch
    val_loss, val_wer = validate(model, val_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val WER: {val_wer:.4f}")
    
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
