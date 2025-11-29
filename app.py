import os
os.environ["HF_HOME"] = "F:/.cache/HuggingFace" 
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


print("Minimal Whisper Fine-Tuning Script")

# STEP 1: Load model 
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None  # Auto-detect language
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# STEP 2: Load dataset 70% train, 20% validation, 10% test
dataset = load_dataset("librispeech_asr", "clean", split="train.100")

# First split: 70% train, 30% temp
train_temp = dataset.train_test_split(test_size=0.30, seed=42)
train_dataset = train_temp["train"]  # 70%

# Second split: temp (30%) -> 20% validation, 10% test
# 20/30 = 0.667 for validation, 10/30 = 0.333 for test
val_test = train_temp["test"].train_test_split(test_size=0.333, seed=42)
val_dataset = val_test["train"]    # 20% of original (used during training)
test_dataset = val_test["test"]    # 10% of original (final evaluation)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# STEP 3: Prepare data (simple preprocessing)
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Cast audio and map preprocessing on each split
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

# STEP 4: Data collator 
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# STEP 5: Training settings (CPU-friendly!)
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8, #look 8 audios at a time
    num_train_epochs=3,
    learning_rate=1e-5, #(0.00001)
    fp16=True,  # GPU
    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=225,

    gradient_accumulation_steps=1,
    save_total_limit=3,
)

# STEP 6: Metric
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    #models output and labels output
    pred_ids = pred.predictions 
    label_ids = pred.label_ids
    # Get out the -100 which we added in datacollector. so we replace with pad token id. to ignore at the time of decoding.
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    #Decode -> Numbers to text REAL TEXT
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# STEP 7: Train!
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Validation set for monitoring during training
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Training...")
trainer.train(resume_from_checkpoint=True)

# STEP 8: Final evaluation on TEST set (unseen data)
print("Final Evaluation on Test Set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test WER: {test_results['eval_wer']:.2%}")

# STEP 9: Save
model.save_pretrained("./whisper-small-finetuned")
processor.save_pretrained("./whisper-small-finetuned")
print("Model saved.")