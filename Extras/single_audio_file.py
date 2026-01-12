from datasets import load_dataset
import soundfile as sf

# Dataset load karo
ds = load_dataset("SPRINGLab/LibriSpeech-100")

# Ek sample lo
sample = ds['train'][0]

# Audio save karo
audio_array = sample['audio']['array']
sample_rate = sample['audio']['sampling_rate']

sf.write("libri_sample.wav", audio_array, sample_rate)

print(f"Text: {sample['text']}")
print(f"Saved: libri_sample.wav")