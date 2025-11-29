import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an audio file
path = "1.mp3"
y, sr = librosa.load(path) 

#Compute the Short-Time Fourier Transform (STFT):
S = librosa.stft(y)


#Convert Amplitude to Decibels:
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

#Visualize the Spectrogram:
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()

mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
