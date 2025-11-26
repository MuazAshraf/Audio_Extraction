from app import pipeline

# Load your fine-tuned model
pipe = pipeline("automatic-speech-recognition", model="./whisper-small-finetuned")

# Test on your audio
audio = "1.mp3"  # Your test file
result = pipe(audio)

print(result["text"])