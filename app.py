import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_AUDIO_DECODER"] = "soundfile"

from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate

print("Whisper Fine-Tuning on GPU")

# Load processor + model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
dataset = load_dataset("librispeech_asr", "clean", split="train.100")

# Split 70/20/10
train_temp = dataset.train_test_split(test_size=0.30, seed=42)
train_dataset = train_temp["train"]
val_test = train_temp["test"].train_test_split(test_size=0.333, seed=42)
val_dataset = val_test["train"]
test_dataset = val_test["test"]

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Preprocess
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Training Arguments (GPU optimized)
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    fp16=True,                       # GPU fast training
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=3,
)

# Metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    preds = pred.predictions
    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test WER: {test_results['eval_wer']:.2%}")

model.save_pretrained("./whisper-small-finetuned")
processor.save_pretrained("./whisper-small-finetuned")
print("Model saved.")