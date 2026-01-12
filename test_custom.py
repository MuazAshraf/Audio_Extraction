import torch
import librosa
import numpy as np
from vocab import Vocabulary
from cnn_encoder import Seq2SeqASR

# Load model
vocab = Vocabulary()
model = Seq2SeqASR(vocab_size=vocab.vocab_size)
model.load_state_dict(torch.load("custom_models/model_final.pt", map_location="cpu"))
model.eval()

# Load audio
audio, sr = librosa.load("1.mp3", sr=16000)

# Mel spectrogram
mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, n_fft=400, hop_length=160)
mel_db = librosa.power_to_db(mel, ref=np.max)
spec = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()

# Transcribe
with torch.no_grad():
    encoder_output = model.encoder(spec)
    hidden = torch.tanh(model.init_hidden(encoder_output.mean(dim=1))).unsqueeze(0)
    input_char = torch.tensor([1])  # <sos>

    result = []
    for _ in range(200):
        context, _ = model.attention(hidden[-1], encoder_output)
        pred, hidden = model.decoder(input_char, hidden, context)
        idx = pred.argmax(1).item()
        if idx == 2:  # <eos>
            break
        if idx > 2:
            result.append(vocab.idx_to_char[idx])
        input_char = torch.tensor([idx])

print("Transcription:", ''.join(result))
