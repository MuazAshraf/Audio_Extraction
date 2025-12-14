from datasets import load_dataset

# cache_dir = r"F:\.cache\HuggingFace\datasets"

# ds = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=cache_dir)
ds = load_dataset("SPRINGLab/LibriSpeech-100")
print(len(ds)) 