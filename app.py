import os
os.environ["HF_AUDIO_DECODER"] = "soundfile"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

print("🚀 Minimal Whisper Fine-Tuning Script")

# STEP 1: Load model (you already have this!)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None  # Auto-detect language
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# STEP 2: Load dataset (works without issues!)
dataset = load_dataset("librispeech_asr", "clean", split="validation")
dataset = dataset.train_test_split(test_size=0.2)

# STEP 3: Prepare data (simple preprocessing)
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

# STEP 4: Data collator Handles all padding/sizing automatically
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
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    fp16=False,  # CPU
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
)

# STEP 6: Metric Word Error Rate
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# STEP 7: Train!
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("⏳ Training...")
trainer.train()

# STEP 8: Save
model.save_pretrained("./whisper-small-finetuned")
processor.save_pretrained("./whisper-small-finetuned")
print("✅ Done! Model saved.")




#Extraction part --pending
# response = client.responses.create(
#     model="gpt-5",
#     reasoning={"effort": "low"},
#     instructions="""Analyze & View the audio Data and carefully without lossing any INORFATION Extract following Information: 
#     1. Name of the speaker 
#     2. Topic of the speech  
#     3. Key Points 
#     4. Conclusion""",
#     input=text_output,
# )

# print(response.output_text)